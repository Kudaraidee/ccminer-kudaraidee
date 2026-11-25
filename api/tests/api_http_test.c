/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Unit tests for the HTTP parser and router in api_http.c.
 *
 * No socket, no miner, no network: api_http_parse() is a pure function over a
 * buffer, which is the whole reason it was written that way. Covers the happy
 * path, every malformed shape the parser has to answer rather than crash on,
 * the size limits, and split reads (the parser must report INCOMPLETE for any
 * prefix of a valid request and OK once the last byte arrives).
 *
 * Build (from the repo root):
 *   cc -I. -Icompat/jansson -o api_http_test api/tests/api_http_test.c api_http.c -ljansson
 * or with the bundled jansson, see api/tests/build-api-http-test.sh
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "core/api/api_http.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...) do { \
	checks++; \
	if (!(cond)) { failures++; printf("  FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static api_parse_status parse(const char *s, api_request *r)
{
	return api_http_parse(s, strlen(s), r);
}

/* ------------------------------------------------------------ happy path -- */

static void test_basic(void)
{
	api_request r;
	printf("basic requests\n");

	CHECK(parse("GET /api/v1/summary HTTP/1.1\r\nHost: x\r\n\r\n", &r) == API_PARSE_OK, "simple GET");
	CHECK(r.method == API_M_GET, "method GET");
	CHECK(strcmp(r.path, "/api/v1/summary") == 0, "path, got '%s'", r.path);
	CHECK(r.query[0] == '\0', "no query");
	CHECK(!r.pretty, "pretty off");

	CHECK(parse("GET /api/v1/threads?id=2&pretty=1 HTTP/1.1\r\n\r\n", &r) == API_PARSE_OK, "query GET");
	CHECK(strcmp(r.path, "/api/v1/threads") == 0, "path split from query, got '%s'", r.path);
	CHECK(strcmp(r.query, "id=2&pretty=1") == 0, "query, got '%s'", r.query);
	CHECK(r.pretty, "pretty=1 detected");

	CHECK(parse("HEAD /api/v1/ HTTP/1.1\r\n\r\n", &r) == API_PARSE_OK, "HEAD");
	CHECK(r.method == API_M_HEAD, "method HEAD");

	CHECK(parse("POST /api/v1/quit HTTP/1.1\r\nContent-Length: 0\r\n\r\n", &r) == API_PARSE_OK, "POST no body");
	CHECK(r.method == API_M_POST && r.body_len == 0, "POST zero length");

	CHECK(parse("POST /api/v1/pools/switch HTTP/1.1\r\nContent-Length: 11\r\n\r\n{\"index\":1}", &r) == API_PARSE_OK,
	      "POST with body");
	CHECK(r.body_len == 11 && strcmp(r.body, "{\"index\":1}") == 0, "body, got '%s'", r.body);
}

/* --------------------------------------------------------------- headers -- */

static void test_headers(void)
{
	api_request r;
	printf("headers\n");

	CHECK(parse("GET / HTTP/1.1\r\nauthorization: bearer SEKRIT\r\n\r\n", &r) == API_PARSE_OK, "ci header");
	CHECK(strcmp(r.auth_bearer, "SEKRIT") == 0, "case-insensitive Authorization, got '%s'", r.auth_bearer);

	CHECK(parse("GET / HTTP/1.1\r\nAuthorization:    Bearer   spaced\r\n\r\n", &r) == API_PARSE_OK, "ws header");
	CHECK(strcmp(r.auth_bearer, "spaced") == 0, "leading whitespace stripped, got '%s'", r.auth_bearer);

	CHECK(parse("GET / HTTP/1.1\r\nAuthorization: Basic abc\r\n\r\n", &r) == API_PARSE_OK, "non-bearer");
	CHECK(r.auth_bearer[0] == '\0', "Basic is not a bearer token");

	CHECK(parse("GET /x HTTP/1.1\r\nUpgrade: websocket\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n", &r)
	      == API_PARSE_OK, "ws upgrade");
	CHECK(r.upgrade_websocket, "upgrade flagged");
	CHECK(strcmp(r.ws_key, "dGhlIHNhbXBsZSBub25jZQ==") == 0, "ws key, got '%s'", r.ws_key);

	CHECK(parse("GET / HTTP/1.1\r\nOrigin: http://h\r\n\r\n", &r) == API_PARSE_OK, "origin");
	CHECK(strcmp(r.origin, "http://h") == 0, "origin captured");
}

/* --------------------------------------------------------------- refusals -- */

static void test_malformed(void)
{
	api_request r;
	printf("malformed input must be answered, not crashed on\n");

	CHECK(parse("GET\r\n\r\n", &r) == API_PARSE_BAD, "no target");
	CHECK(parse("GET /x\r\n\r\n", &r) == API_PARSE_BAD, "no version");
	CHECK(parse("GET relative HTTP/1.1\r\n\r\n", &r) == API_PARSE_BAD, "target must be absolute path");
	CHECK(parse("GET /a/../b HTTP/1.1\r\n\r\n", &r) == API_PARSE_BAD, "dot-dot rejected");
	CHECK(parse("GET /a%2e%2e/b HTTP/1.1\r\n\r\n", &r) == API_PARSE_BAD, "encoded dot-dot rejected");
	CHECK(parse("GET /a%00b HTTP/1.1\r\n\r\n", &r) == API_PARSE_BAD, "%%00 rejected");
	CHECK(parse("GET /a%zz HTTP/1.1\r\n\r\n", &r) == API_PARSE_BAD, "bad escape rejected");
	CHECK(parse("GET / HTTP/1.1\r\nContent-Length: abc\r\n\r\n", &r) == API_PARSE_BAD, "non-numeric length");
	CHECK(parse("GET / HTTP/1.1\r\nContent-Length: -5\r\n\r\n", &r) == API_PARSE_BAD, "negative length");
	CHECK(parse("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", &r) == API_PARSE_UNSUPPORTED,
	      "chunked is 501, never guessed at");
	CHECK(parse("POST / HTTP/1.1\r\n\r\nbody-without-length", &r) == API_PARSE_BAD,
	      "undelimited body rejected rather than truncated");

	/* percent-decoding is applied and legal escapes survive */
	CHECK(parse("GET /api/v1/a%2Fb HTTP/1.1\r\n\r\n", &r) == API_PARSE_OK, "legal escape");
	CHECK(strcmp(r.path, "/api/v1/a/b") == 0, "decoded, got '%s'", r.path);
}

static void test_limits(void)
{
	printf("limits\n");
	api_request r;
	char *big = malloc(API_HTTP_MAX_REQLINE + API_HTTP_MAX_HEADER_BYTES + API_HTTP_MAX_BODY + 4096);

	/* over-long request line */
	int n = sprintf(big, "GET /");
	for (int i = 0; i < API_HTTP_MAX_REQLINE + 10; i++) big[n++] = 'a';
	n += sprintf(big + n, " HTTP/1.1\r\n\r\n");
	CHECK(api_http_parse(big, (size_t) n, &r) == API_PARSE_TOO_LARGE, "long request line");

	/* too many header lines */
	n = sprintf(big, "GET / HTTP/1.1\r\n");
	for (int i = 0; i < API_HTTP_MAX_HEADER_LINES + 5; i++)
		n += sprintf(big + n, "X-H%d: v\r\n", i);
	n += sprintf(big + n, "\r\n");
	CHECK(api_http_parse(big, (size_t) n, &r) == API_PARSE_TOO_LARGE, "too many headers");

	/* body over the cap */
	n = sprintf(big, "POST / HTTP/1.1\r\nContent-Length: %d\r\n\r\n", API_HTTP_MAX_BODY + 1);
	CHECK(api_http_parse(big, (size_t) n, &r) == API_PARSE_TOO_LARGE, "oversized body");

	free(big);
}

/* ------------------------------------------------------------ split reads -- */

static void test_split_reads(void)
{
	printf("split reads: every prefix is INCOMPLETE, the whole thing is OK\n");
	const char *full = "POST /api/v1/pools/url HTTP/1.1\r\n"
	                   "Content-Type: application/json\r\n"
	                   "Content-Length: 13\r\n"
	                   "\r\n"
	                   "{\"url\":\"x\"}!!";
	size_t total = strlen(full);
	api_request r;
	int bad = 0;

	for (size_t i = 1; i < total; i++) {
		api_parse_status st = api_http_parse(full, i, &r);
		if (st != API_PARSE_INCOMPLETE) {
			bad++;
			printf("  prefix of %zu bytes returned %d, expected INCOMPLETE\n", i, (int) st);
		}
	}
	CHECK(bad == 0, "%d prefixes mis-parsed", bad);
	CHECK(api_http_parse(full, total, &r) == API_PARSE_OK, "complete request parses");
	CHECK(r.body_len == 13, "body length %zu", r.body_len);
}

/* ---------------------------------------------------------------- routing -- */

static int dummy(const api_request *q, void *c, json_t **o, char *e, size_t n)
{
	(void) q; (void) c; (void) o; (void) e; (void) n; return 200;
}

static void test_routing(void)
{
	printf("routing\n");
	static const api_route routes[] = {
		{ API_M_GET,  "/api/v1/summary",      API_PRIV_READ,    true,  dummy },
		{ API_M_GET,  "/api/v1/history",      API_PRIV_READ,    false, dummy },  /* 501 side */
		{ API_M_GET,  "/api/v1/pools/",       API_PRIV_READ,    true,  dummy },  /* prefix   */
		{ API_M_POST, "/api/v1/pools/switch", API_PRIV_WRITE,   true,  dummy },
		{ API_M_POST, "/api/v1/control/stop", API_PRIV_CONTROL, true,  dummy },
	};
	const size_t n = sizeof(routes) / sizeof(routes[0]);
	api_request r;
	int status;

	parse("GET /api/v1/summary HTTP/1.1\r\n\r\n", &r);
	CHECK(api_http_route(routes, n, &r, &status) == &routes[0] && status == 200, "exact match");

	parse("HEAD /api/v1/summary HTTP/1.1\r\n\r\n", &r);
	CHECK(api_http_route(routes, n, &r, &status) == &routes[0] && status == 200, "HEAD uses the GET route");

	parse("GET /api/v1/nope HTTP/1.1\r\n\r\n", &r);
	CHECK(api_http_route(routes, n, &r, &status) == NULL && status == 404, "unknown path is 404");

	parse("POST /api/v1/summary HTTP/1.1\r\nContent-Length: 0\r\n\r\n", &r);
	CHECK(api_http_route(routes, n, &r, &status) == NULL && status == 405,
	      "known path + wrong verb is 405, not 404 (got %d)", status);

	parse("GET /api/v1/history HTTP/1.1\r\n\r\n", &r);
	CHECK(api_http_route(routes, n, &r, &status) == NULL && status == 501,
	      "unavailable route is 501 (got %d)", status);

	parse("GET /api/v1/pools/3 HTTP/1.1\r\n\r\n", &r);
	CHECK(api_http_route(routes, n, &r, &status) == &routes[2] && status == 200, "prefix route matches");

	/* Trailing slashes are tolerated and normalised away, so a bare prefix
	 * resolves to the collection route rather than the {id} route. That is what
	 * makes GET /api/v1/ reach the index rather than 404. */
	parse("GET /api/v1/pools/ HTTP/1.1\r\n\r\n", &r);
	CHECK(strcmp(r.path, "/api/v1/pools") == 0, "trailing slash stripped, got '%s'", r.path);
	CHECK(api_http_route(routes, n, &r, &status) == NULL && status == 404,
	      "/pools/ no longer matches the {n} prefix route");

	parse("GET /api/v1/summary/// HTTP/1.1\r\n\r\n", &r);
	CHECK(strcmp(r.path, "/api/v1/summary") == 0, "repeated trailing slashes stripped");
	CHECK(api_http_route(routes, n, &r, &status) == &routes[0], "and still routes");
}

static void test_error_bodies(void)
{
	printf("error bodies\n");
	CHECK(strcmp(api_http_error_code(404), "not_found") == 0, "404 code");
	CHECK(strcmp(api_http_error_code(501), "not_implemented") == 0, "501 code");
	CHECK(strcmp(api_http_error_code(999), "internal_error") == 0, "unknown status falls back");

	api_http_config cfg;
	memset(&cfg, 0, sizeof(cfg));
	cfg.miner_json = "{\"name\":\"x\",\"kind\":\"gpu\"}";
	char *b = api_http_error_body(&cfg, 404, "nope");
	CHECK(b != NULL, "body built");
	if (b) {
		CHECK(strstr(b, "\"code\": \"not_found\"") || strstr(b, "\"code\":\"not_found\""), "code present: %s", b);
		CHECK(strstr(b, "\"status\": 404") || strstr(b, "\"status\":404"), "status present");
		CHECK(strstr(b, "\"miner\"") != NULL, "miner envelope present on errors");
		free(b);
	}
}

int main(void)
{
	printf("api_http parser/router tests\n\n");
	test_basic();
	test_headers();
	test_malformed();
	test_limits();
	test_split_reads();
	test_routing();
	test_error_bodies();
	printf("\n%d checks, %d failures\n", checks, failures);
	return failures ? 1 : 0;
}
