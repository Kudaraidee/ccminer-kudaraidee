#ifndef SECP256K1_MODINV32_DEVICE_CUH
#define SECP256K1_MODINV32_DEVICE_CUH

/*
 * Modular inverse mod p by safegcd (Bernstein-Yang) divsteps.
 *
 * Device transcription of modern libsecp256k1's modinv32_impl.h plus
 * field_10x26_impl.h's signed30 conversions (MIT, (c) 2020 Peter Dettman).
 * Replaces the p-2 addition chain, which is all the vendored 2015 tree has.
 *
 * Only the CONSTANT-TIME path is ported - the opposite of the better choice on a
 * CPU. Upstream's _var variants are faster there, but their trip counts are
 * data-dependent (trailing zeros, a shrinking limb count), which on a GPU is warp
 * divergence; the fixed 20 x 30 = 600 divsteps are uniform across all 32 lanes.
 *
 * Upstream threads a modinv32_modinfo pointer through every function so one
 * implementation can serve both the field and the scalar. curvehash only inverts
 * mod p, so the modulus limbs and 1/p mod 2^30 are function-local constants with
 * upstream's values - same arithmetic, but nvcc folds them, and six of p's nine
 * signed30 limbs are zero so most of update_de_30's modulus products vanish.
 *
 * Constant-time in x, but NOT usable for secret keys: the surrounding curvehash
 * kernel is variable-time by design (see the ecmult_gen header).
 *
 * Requires secp256k1_field_device.cuh (secp256k1_fe, secp256k1_fe_normalize).
 */

/* p = 2^256 - 2^32 - 977 as nine signed 30-bit limbs, and -1/p mod 2^30.
 * == upstream secp256k1_const_modinfo_fe. */
#define CH_MODINV_P_LIMBS  { -0x3D1, -4, 0, 0, 0, 0, 0, 0, 65536 }
#define CH_MODINV_P_INV30  ((uint32_t)0x2DDACACFu)

/* 256-bit integer as nine signed 30-bit limbs; value = sum(v[i] * 2^(30*i)). */
typedef struct { int32_t v[9]; } secp256k1_modinv32_signed30;

/* Transition matrix t = [[u, v], [q, r]] (section 3 of upstream's explanation). */
typedef struct { int32_t u, v, q, r; } secp256k1_modinv32_trans2x2;

/* Replace r with r + modulus if r is negative, negate if sign < 0, and bring it
 * to range [0,modulus). Input limbs in (-2^30,2^30); output limbs in [0,2^30). */
__device__ static void secp256k1_modinv32_normalize_30(secp256k1_modinv32_signed30 *r, int32_t sign)
{
    const int32_t M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t P[9] = CH_MODINV_P_LIMBS;
    int32_t r0 = r->v[0], r1 = r->v[1], r2 = r->v[2], r3 = r->v[3], r4 = r->v[4],
            r5 = r->v[5], r6 = r->v[6], r7 = r->v[7], r8 = r->v[8];
    int32_t cond_add, cond_negate;

    /* Add the modulus if negative, then negate if requested: brings r from
     * (-2*modulus,modulus) to (-modulus,modulus). Cannot overflow int32. */
    cond_add = r8 >> 31;
    r0 += P[0] & cond_add;
    r1 += P[1] & cond_add;
    r2 += P[2] & cond_add;
    r3 += P[3] & cond_add;
    r4 += P[4] & cond_add;
    r5 += P[5] & cond_add;
    r6 += P[6] & cond_add;
    r7 += P[7] & cond_add;
    r8 += P[8] & cond_add;
    cond_negate = sign >> 31;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    r5 = (r5 ^ cond_negate) - cond_negate;
    r6 = (r6 ^ cond_negate) - cond_negate;
    r7 = (r7 ^ cond_negate) - cond_negate;
    r8 = (r8 ^ cond_negate) - cond_negate;
    /* Propagate the top bits, bringing limbs back to (-2^30,2^30). */
    r1 += r0 >> 30; r0 &= M30;
    r2 += r1 >> 30; r1 &= M30;
    r3 += r2 >> 30; r2 &= M30;
    r4 += r3 >> 30; r3 &= M30;
    r5 += r4 >> 30; r4 &= M30;
    r6 += r5 >> 30; r5 &= M30;
    r7 += r6 >> 30; r6 &= M30;
    r8 += r7 >> 30; r7 &= M30;

    /* Add the modulus again if still negative, bringing r to [0,modulus). */
    cond_add = r8 >> 31;
    r0 += P[0] & cond_add;
    r1 += P[1] & cond_add;
    r2 += P[2] & cond_add;
    r3 += P[3] & cond_add;
    r4 += P[4] & cond_add;
    r5 += P[5] & cond_add;
    r6 += P[6] & cond_add;
    r7 += P[7] & cond_add;
    r8 += P[8] & cond_add;
    /* And propagate again. */
    r1 += r0 >> 30; r0 &= M30;
    r2 += r1 >> 30; r1 &= M30;
    r3 += r2 >> 30; r2 &= M30;
    r4 += r3 >> 30; r3 &= M30;
    r5 += r4 >> 30; r4 &= M30;
    r6 += r5 >> 30; r5 &= M30;
    r7 += r6 >> 30; r6 &= M30;
    r8 += r7 >> 30; r7 &= M30;

    r->v[0] = r0; r->v[1] = r1; r->v[2] = r2; r->v[3] = r3; r->v[4] = r4;
    r->v[5] = r5; r->v[6] = r6; r->v[7] = r7; r->v[8] = r8;
}

/* Compute the transition matrix and zeta for 30 divsteps.
 * Input:  zeta, and the bottom limbs f0, g0 of f and g. Output: t.
 * Return: final zeta.  (divsteps_n_matrix in upstream's explanation.) */
__device__ static int32_t secp256k1_modinv32_divsteps_30(int32_t zeta, uint32_t f0, uint32_t g0,
                                                         secp256k1_modinv32_trans2x2 *t)
{
    /* u,v,q,r are semantically signed in [-2^30,2^30] but held as unsigned mod
     * 2^32 so that left shifting is defined. */
    uint32_t u = 1, v = 0, q = 0, r = 1;
    uint32_t c1, c2, f = f0, g = g0, x, y, z;
    int i;

    #pragma unroll 1
    for (i = 0; i < 30; ++i) {
        /* Conditional masks for (zeta < 0) and for (g & 1). */
        c1 = (uint32_t)(zeta >> 31);
        c2 = -(g & 1);
        /* x,y,z: conditionally negated f,u,v. */
        x = (f ^ c1) - c1;
        y = (u ^ c1) - c1;
        z = (v ^ c1) - c1;
        /* Conditionally add x,y,z to g,q,r. */
        g += x & c2;
        q += y & c2;
        r += z & c2;
        /* c1 now masks (zeta < 0) AND (g & 1). */
        c1 &= c2;
        /* Conditionally change zeta into -zeta-2 or zeta-1. */
        zeta = (int32_t)(((uint32_t)zeta ^ c1) - 1u);
        /* Conditionally add g,q,r to f,u,v. */
        f += g & c1;
        u += q & c1;
        v += r & c1;
        /* Shifts. */
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }
    t->u = (int32_t)u;
    t->v = (int32_t)v;
    t->q = (int32_t)q;
    t->r = (int32_t)r;
    return zeta;
}

/* d,e <- (t/2^30) * [d,e] mod modulus. On input and output d,e are in
 * (-2*modulus,modulus); all output limbs are in (-2^30,2^30). */
__device__ static void secp256k1_modinv32_update_de_30(secp256k1_modinv32_signed30 *d,
                                                       secp256k1_modinv32_signed30 *e,
                                                       const secp256k1_modinv32_trans2x2 *t)
{
    const int32_t M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t P[9] = CH_MODINV_P_LIMBS;
    const uint32_t PINV30 = CH_MODINV_P_INV30;
    const int32_t u = t->u, v = t->v, q = t->q, r = t->r;
    int32_t di, ei, md, me, sd, se;
    int64_t cd, ce;
    int i;

    /* [md,me] start at zero; plus [u,q] if d is negative; plus [v,r] if e is. */
    sd = d->v[8] >> 31;
    se = e->v[8] >> 31;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);
    /* Begin computing t*[d,e]. */
    di = d->v[0];
    ei = e->v[0];
    cd = (int64_t)u * di + (int64_t)v * ei;
    ce = (int64_t)q * di + (int64_t)r * ei;
    /* Correct md,me so that t*[d,e]+modulus*[md,me] has 30 zero bottom bits. */
    md -= (PINV30 * (uint32_t)cd + md) & M30;
    me -= (PINV30 * (uint32_t)ce + me) & M30;
    /* Update the start of t*[d,e]+modulus*[md,me] now md,me are known. */
    cd += (int64_t)P[0] * md;
    ce += (int64_t)P[0] * me;
    /* The low 30 bits are zero by construction; throw them away. */
    cd >>= 30;
    ce >>= 30;
    /* Limbs 1..8, stored into output limb i-1 (shifted down by 30 bits). */
    #pragma unroll
    for (i = 1; i < 9; ++i) {
        di = d->v[i];
        ei = e->v[i];
        cd += (int64_t)u * di + (int64_t)v * ei;
        ce += (int64_t)q * di + (int64_t)r * ei;
        cd += (int64_t)P[i] * md;   /* P[2..7] == 0: folded away by nvcc */
        ce += (int64_t)P[i] * me;
        d->v[i - 1] = (int32_t)cd & M30; cd >>= 30;
        e->v[i - 1] = (int32_t)ce & M30; ce >>= 30;
    }
    /* What remains is limb 9; store it as output limb 8. */
    d->v[8] = (int32_t)cd;
    e->v[8] = (int32_t)ce;
}

/* f,g <- (t/2^30) * [f,g]. */
__device__ static void secp256k1_modinv32_update_fg_30(secp256k1_modinv32_signed30 *f,
                                                       secp256k1_modinv32_signed30 *g,
                                                       const secp256k1_modinv32_trans2x2 *t)
{
    const int32_t M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t u = t->u, v = t->v, q = t->q, r = t->r;
    int32_t fi, gi;
    int64_t cf, cg;
    int i;

    fi = f->v[0];
    gi = g->v[0];
    cf = (int64_t)u * fi + (int64_t)v * gi;
    cg = (int64_t)q * fi + (int64_t)r * gi;
    /* Bottom 30 bits are zero by construction; throw them away. */
    cf >>= 30;
    cg >>= 30;
    #pragma unroll
    for (i = 1; i < 9; ++i) {
        fi = f->v[i];
        gi = g->v[i];
        cf += (int64_t)u * fi + (int64_t)v * gi;
        cg += (int64_t)q * fi + (int64_t)r * gi;
        f->v[i - 1] = (int32_t)cf & M30; cf >>= 30;
        g->v[i - 1] = (int32_t)cg & M30; cg >>= 30;
    }
    f->v[8] = (int32_t)cf;
    g->v[8] = (int32_t)cg;
}

/* x <- 1/x mod p (constant time in x). x must be in [0,p); x == 0 yields 0. */
__device__ static void secp256k1_modinv32(secp256k1_modinv32_signed30 *x)
{
    const int32_t P[9] = CH_MODINV_P_LIMBS;
    /* d=0, e=1, f=modulus, g=x, zeta=-1 (zeta = -(delta+1/2), delta = 1/2). */
    secp256k1_modinv32_signed30 d, e, f, g;
    int i;
    int32_t zeta = -1;

    #pragma unroll
    for (i = 0; i < 9; ++i) { d.v[i] = 0; e.v[i] = 0; f.v[i] = P[i]; }
    e.v[0] = 1;
    g = *x;

    /* 20 iterations of 30 divsteps = 600; 590 suffices for 256-bit inputs. */
    #pragma unroll 1
    for (i = 0; i < 20; ++i) {
        secp256k1_modinv32_trans2x2 t;
        zeta = secp256k1_modinv32_divsteps_30(zeta, (uint32_t)f.v[0], (uint32_t)g.v[0], &t);
        secp256k1_modinv32_update_de_30(&d, &e, &t);
        secp256k1_modinv32_update_fg_30(&f, &g, &t);
    }

    /* g is now 0, and (unless x was 0) f == +/-1 with d == +/- the inverse. */
    secp256k1_modinv32_normalize_30(&d, f.v[8]);
    *x = d;
}

/* --- secp256k1_fe <-> signed30, from modern field_10x26_impl.h --- */

__device__ static void secp256k1_fe_from_signed30(secp256k1_fe *r, const secp256k1_modinv32_signed30 *a)
{
    const uint32_t M26 = UINT32_MAX >> 6;
    const uint32_t a0 = a->v[0], a1 = a->v[1], a2 = a->v[2], a3 = a->v[3], a4 = a->v[4],
                   a5 = a->v[5], a6 = a->v[6], a7 = a->v[7], a8 = a->v[8];

    r->n[0] =  a0                   & M26;
    r->n[1] = (a0 >> 26 | a1 <<  4) & M26;
    r->n[2] = (a1 >> 22 | a2 <<  8) & M26;
    r->n[3] = (a2 >> 18 | a3 << 12) & M26;
    r->n[4] = (a3 >> 14 | a4 << 16) & M26;
    r->n[5] = (a4 >> 10 | a5 << 20) & M26;
    r->n[6] = (a5 >>  6 | a6 << 24) & M26;
    r->n[7] = (a6 >>  2           ) & M26;
    r->n[8] = (a6 >> 28 | a7 <<  2) & M26;
    r->n[9] = (a7 >> 24 | a8 <<  6);
}

/* Input must be NORMALIZED (limbs canonical and value < p). */
__device__ static void secp256k1_fe_to_signed30(secp256k1_modinv32_signed30 *r, const secp256k1_fe *a)
{
    const uint32_t M30 = UINT32_MAX >> 2;
    const uint64_t a0 = a->n[0], a1 = a->n[1], a2 = a->n[2], a3 = a->n[3], a4 = a->n[4],
                   a5 = a->n[5], a6 = a->n[6], a7 = a->n[7], a8 = a->n[8], a9 = a->n[9];

    r->v[0] = (int32_t)((a0       | a1 << 26) & M30);
    r->v[1] = (int32_t)((a1 >>  4 | a2 << 22) & M30);
    r->v[2] = (int32_t)((a2 >>  8 | a3 << 18) & M30);
    r->v[3] = (int32_t)((a3 >> 12 | a4 << 14) & M30);
    r->v[4] = (int32_t)((a4 >> 16 | a5 << 10) & M30);
    r->v[5] = (int32_t)((a5 >> 20 | a6 <<  6) & M30);
    r->v[6] = (int32_t)((a6 >> 24 | a7 <<  2
                                 | a8 << 28) & M30);
    r->v[7] = (int32_t)((a8 >>  2 | a9 << 24) & M30);
    r->v[8] = (int32_t)( a9 >>  6);
}

/* r = 1/a mod p (and 0 if a == 0, matching the addition chain's a^(p-2)). */
__device__ static void secp256k1_fe_inv_safegcd(secp256k1_fe *r, const secp256k1_fe *a)
{
    secp256k1_fe tmp = *a;
    secp256k1_modinv32_signed30 s;

    secp256k1_fe_normalize(&tmp);      /* to_signed30 requires a normalized input */
    secp256k1_fe_to_signed30(&s, &tmp);
    secp256k1_modinv32(&s);
    secp256k1_fe_from_signed30(r, &s);
}

#endif /* SECP256K1_MODINV32_DEVICE_CUH */
