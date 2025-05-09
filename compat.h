#ifndef __COMPAT_H__
#define __COMPAT_H__

#ifdef WIN32

#include <windows.h>
#include <time.h>

#define localtime_r(src, dst) localtime_s(dst, src)

static __inline void sleep(int secs)
{
	Sleep(secs * 1000);
}

enum {
	PRIO_PROCESS = 0,
};

extern int opt_priority;

static __inline int setpriority(int which, int who, int prio)
{
	switch (opt_priority) {
		case 5:
			prio = THREAD_PRIORITY_TIME_CRITICAL;
			break;
		case 4:
			prio = THREAD_PRIORITY_HIGHEST;
			break;
		case 3:
			prio = THREAD_PRIORITY_ABOVE_NORMAL;
			break;
		case 2:
			prio = THREAD_PRIORITY_NORMAL;
			break;
		case 1:
			prio = THREAD_PRIORITY_BELOW_NORMAL;
			break;
		case 0:
		default:
			prio = THREAD_PRIORITY_IDLE;
	}
	return -!SetThreadPriority(GetCurrentThread(), prio);
}

#ifdef _MSC_VER
#define snprintf(...) _snprintf(__VA_ARGS__)
#define strdup(...) _strdup(__VA_ARGS__)
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#define strcasecmp(x,y) _stricmp(x,y)
typedef int ssize_t;

__inline int msver(void) {
	if (_MSC_VER >= 1930) return 2022;
	if (_MSC_VER >= 1920) return 2019;
	if (_MSC_VER >= 1910) return 2017;
	if (_MSC_VER >= 1900) return 2015;
	if (_MSC_VER >= 1800) return 2013;
	if (_MSC_VER >= 1700) return 2012;
	if (_MSC_VER >= 1600) return 2010;
	if (_MSC_VER >= 1500) return 2008;
	return (_MSC_VER / 100) + 2000;
}

#include <stdlib.h>
static __inline char * dirname(char *file) {
	char buffer[_MAX_PATH] = { 0 };
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
	_splitpath_s(file, drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);
	sprintf(buffer, "%s%s", drive, dir);
	return strdup(buffer);
}
#endif

#endif /* WIN32 */

#ifdef _MSC_VER
# define __func__ __FUNCTION__
# define __thread __declspec(thread)
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
/* dirname() for linux/mingw */
#include <libgen.h>
#endif

#ifndef WIN32
#define MAX_PATH PATH_MAX
#endif

#endif /* __COMPAT_H__ */
