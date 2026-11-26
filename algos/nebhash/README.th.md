# Nebhash (NEB, `neb-pool/1`) — `-a nebhash` / `-a neb` (ภาษาไทย)

[English version](README.md)

เป็น proof of work แบบใช้จำนวนเต็มล้วนที่ NEB ใช้ ทุกครั้งที่ลองขุดจะสร้าง
ChaCha8 key จาก header ขนาด 80 ไบต์ด้วย BLAKE3, สุ่มตัวอย่าง 16 ตัวจากชุดข้อมูล
ที่ตรึงไว้, คำนวณ training gradient ของโมเดล convolutional บนตัวอย่างเหล่านั้น,
ตัดค่า (clip) และรวมผลเป็น ticket ขนาด 256 บิตแบบ little-endian แล้วเปรียบเทียบ
กับ share target ของ pool ไม่มี floating point ไม่มี DAG และไม่ต้องสร้าง epoch ใหม่

`-a neb` เป็น alias ของ `-a nebhash` รองรับทั้ง `stratum+tcp://` แบบข้อความ
ธรรมดาและ TLS แบบ `stratum+ssl://` / `stratum+tcps://`

## ชุดข้อมูล (Dataset)

ชุดข้อมูลถูกล็อกด้วย consensus ให้ดาวน์โหลด `neb-datasets.bin` จากหน้า release
ของ NEB แล้วระบุด้วย `--neb-dataset=PATH` (ค่าเริ่มต้นคือ `neb-datasets.bin` ใน
working directory) ไฟล์ทางการมีขนาด 16,087,037 ไบต์:

```text
SHA256  57e22a4b8aacecb2ebae62f76307ddccb04198ed5c4ba15c1dee90ca7a7c2174
BLAKE3  790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72
```

ตอนเริ่มต้น miner จะคำนวณ BLAKE3 ของไฟล์ทั้งไฟล์แล้วเทียบกับ `dataset_hash` ที่
ส่งมาใน `mining.subscribe` หากไม่ตรงจะหยุดการขุดด้วยข้อความ
`Nebhash: pool dataset hash does not match the local file` ไฟล์นี้เป็นทรัพยากร
ของผู้ใช้จึงไม่ได้รวมอยู่ใน repository ภายในไฟล์มีสอง slot โดย slot 0 คือ
architecture 1 (KEP1, 784 inputs, 26,000 weights)

## โครงสร้างไฟล์

| ไฟล์ | บทบาท |
| :-- | :-- |
| `nebhash.cpp` / `nebhash.h` | สะพานเชื่อม ccminer: โหลดชุดข้อมูล, สร้าง header, เชื่อมการสแกน, ตรวจซ้ำบน host และส่งงาน |
| `nebhash2.c` / `nebhash2.h` | C reference ที่เป็นแหล่งความจริงสำหรับ nebhash v2 (KEP1/RET2): ตั้งค่าโมเดล, gradient ของตัวอย่าง, ประเมิน share, ticket, model step และ state hash |
| `nebhash2_table_cuda.cu` / `.h` | CUDA kernel ที่ miner ใช้จริง: one block ต่อ nonce, ตาราง gradient ต่อตัวอย่าง, ใช้ `dp4a` สำหรับ linear layer |
| `nebhash2_cuda.cu` / `.h` | kernel แบบ forward/backward ตรง ๆ รุ่นก่อนหน้า เก็บไว้สำหรับการเริ่มต้นและอ้างอิง miner ไม่ได้เรียกใช้ |
| `shares.c` / `shares.h` | reference แบบพกพาของ PoW training-share รุ่นเดิม (v1) และค่าคงที่ร่วม (`NEB_NB=16`, `NEB_CLIP=256`, ...) |
| `cuda/blake3_device.cuh` | BLAKE3 กลางสำหรับ CUDA kernel ทั้งสองตัวและ C reference ฝั่ง host ใช้ chunk tree เต็มรูปแบบสำหรับ hash ชุดข้อมูลและโมเดล |
| `chacha8.c` | ChaCha8 แบบพกพาที่ C reference ใช้ |
| `LICENSE.NEB` | ใบอนุญาต Apache-2.0 สำหรับโค้ด reference ที่นำมารวมไว้ |

`NEB_SCAN_CAP` (8192 nonces ต่อ `scan_run`) ช่วยให้หน้าต่างการสแกนอยู่ภายใน
อายุของงาน ขณะเดียวกันก็ลดความถี่ในการซิงก์ระหว่าง host กับ device รายการ hit
มีขนาด `MAX_NONCES + 1` รายการ

สำหรับ linear layer kernel จะจัดเก็บเฉพาะ error words `4 × cout` ไว้ใน shared
memory แล้วโหลด input words `4 × cin` จากแถวของตารางใหม่ต่อ input element
หนึ่งครั้ง ดัชนีแถวทั้ง 16 ตัวถูกแปลงเป็น row offset แบบ 32 บิตเพียงครั้งเดียวต่อ
block (ตัว build จะปฏิเสธตารางที่ใหญ่เกิน 4 GiB) จึงตัดการคูณ 64 บิตออกจากทุก
การโหลดไบต์ของแถว ข้อมูลในแถวทั้ง linear factor และ convolution weight อ่านผ่าน
read-only cache ด้วย `__ldg` และ `THREADS` เท่ากับ 128 จึงมี block พอดีบน SM
ได้สองเท่าที่จำนวน thread เท่าเดิม ตัว accumulator ของ lane ทั้งแปดถูกรวมต่อ
warp แล้ว thread 0 รวมอีกครั้ง แทนการใช้ shared-memory atomic 256 ครั้งต่อ lane
ผลบวกทั้งหมดเป็น modulo `2^32` ดังนั้นการเปลี่ยนลำดับการบวกไม่ทำให้ ticket เปลี่ยน

ตัวสแกนเก็บค่า BLAKE3 chaining value ของ header 64 ไบต์แรกไว้ใช้ซ้ำ
และคำนวณใหม่เมื่อข้อมูลส่วนนี้เปลี่ยน แต่ละ nonce ใช้ 16 ไบต์ท้ายของ header
เพื่อคำนวณ key ให้เสร็จ

ในแต่ละ linear layer ตัวสแกนข้าม output ที่ error factor เป็นศูนย์ครบทั้ง
16 ตัวอย่างที่เลือก โดยเริ่มจากผลรวมใน digest ของ linear gradient ที่เป็นศูนย์
ทุกค่า แล้วบวกส่วนต่างเฉพาะ output ที่ยังต้องคำนวณ การบวกแบบ modulo
`2^32` ทำให้ได้ ticket เดิม

ใน KEP1 และ RET2 ทั้งความกว้างอินพุตของ linear layer (`cin`) และขนาด
CUDA block เป็นพหุคูณของแปด ดัชนีน้ำหนักคือ `f = o * cin + q` และแต่ละ
thread ไล่ดัชนีอินพุต `q` จาก `threadIdx.x` ทีละขนาด block ดังนั้น
`f & 7` จึงเท่ากับ `threadIdx.x & 7` สำหรับ thread นั้นทุกครั้ง
แต่ละ thread สะสมผลจาก linear layer ไว้ในตัวแปร unsigned 32 บิตตัวเดียว
แล้วบวกเข้าช่อง digest ก่อนรวมผลระดับ warp ส่วน convolution ยังสะสมผล
แยกตามช่องเดิม ตัวสแกนตรวจเงื่อนไข `cin` ตามสถาปัตยกรรม และตรวจขนาด
block ตอนคอมไพล์

## ที่มาของโค้ด (Provenance)

`nebhash2.{c,h}`, `shares.{c,h}`, `chacha8.{c,h}` และ table
kernel ที่พัฒนาจาก reference ถูกนำมารวมไว้จากการใช้งานของ NEB ภายใต้
Apache-2.0 (`LICENSE.NEB`) ส่วนสะพาน ccminer และ stratum ใช้ GPLv2 ของโปรเจกต์

## Stratum (`neb-pool/1`)

| ข้อความ | รูปแบบ |
| :-- | :-- |
| `mining.subscribe` | `["<user agent>", "neb-pool/1"]` → object ที่มี `session`, `extranonce1`, `extranonce2_size`, `chain_id`, `dataset_hash` |
| `mining.set_target` | `["<32-byte LE hex>"]` — ขีดจำกัดของ share |
| `mining.set_model` | `[slot, "<weights_hash hex>", "<packed weights hex>"]` |
| `mining.notify` | job object เดียว: `job_id`, `version`, `parent`, `time_start`, `bits`, `target`, `share_target`, `coinbase_prefix/suffix`, `branch`, `tx_count`, `shares_root`, `height`, `slot`, `weights_hash`, `expiry` |
| `mining.submit` | `[login, job_id, extranonce2_hex, ntime(number), nonce(number)]` |
| `mining.block` | ส่วนเสริมจาก pool ไม่บังคับ ส่งเฉพาะหลัง node ยอมรับ candidate โดย params จะเป็น `["<job_id>", "<nonce>"]` หรือ object ที่มี `job_id`, `extranonce2`, `ntime`, `nonce`, `blockhash`, `height` จากนั้น ccminer จะ log `yay!!! ... solved: N` |

ฟิลด์ target ใน notify อธิบายสถานะของ chain/block ส่วนค่าที่ตัดสินว่าเป็น share
คือ `mining.set_target` การแปลง target เป็น difficulty ใช้สูตรของ yiimp
`2^256 / (target + 1)` ตัวเลือก `sd=` ในรหัสผ่านถูกตีความโดย pool
(yiimp ใช้ scale `sd * 1e10`) miner ไม่ได้ parse ค่านี้เลย

## ความถูกต้อง

- kernel ตารางบน GPU ถูกตรวจเทียบกับ C reference ด้วยชุดทดสอบ differential
  แบบเทียบ ticket ทุกตัวบน `sm_61`: 384 nonces, ห้าขอบเขต rank และ ticket ที่
  ตรงเป๊ะแปดตัวต่อกรณี สำหรับโมเดลทั้งสอง architecture และน้ำหนักแบบ
  initial / trained / all-7 / all-(-8) / alternating ทุกกรณีผ่าน
- ทุกครั้งที่ GPU พบ hit จะถูกคำนวณซ้ำบน host ด้วย `neb2_share_eval()` ก่อน
  `mining.submit` ดังนั้นข้อผิดพลาดของ kernel อาจทำให้พลาด share ในเครื่อง
  แต่ไม่มีทางส่ง share ผิดไปยัง pool
- `NEB_NEGATIVE_CONTROL_TABLE_CLIP` ตั้งค่า clip ให้ต่ำกว่า `NEB_CLIP` หนึ่ง
  เพื่อแสดงให้เห็นว่าการทดสอบ differential ตรวจพบ gradient ที่ผิด

## การใช้งาน

```bash
ccminer -a nebhash \
  -o stratum+ssl://pool.example.org:3645 \
  -u <neb-address>.<worker> -p c=NEB \
  --neb-dataset=/path/to/neb-datasets.bin
```

`stratum+tcps://` ใช้เป็น alias ของ `stratum+ssl://` ได้ การตรวจสอบใบรับรอง
TLS ยังเปิดอยู่เสมอ ใช้ `--cert` เฉพาะเพื่อระบุ CA bundle ที่กำหนดเอง

## บันทึกความเร็ว

| GPU | Arch | อัตรา | หมายเหตุ |
| :-- | :-- | :-- | :-- |
| GTX 1050 Ti | sm_61 | เฉลี่ย ~0.43 MH/s (~0.25–0.55) | 2026-11-26, ขุดกับ pool จริง, ใช้ชุดข้อมูลทางการ, kernel `sm_61` รัน 5 นาทีได้ 432.33 kH/s, share 1/1 ถูกยอมรับ, ไม่มี reject รัน A/B 150 วิได้เฉลี่ย 493.47 kH/s รันก่อนหน้ายอมรับทุก share ทั้งที่มีและไม่มี `sd=` |
| GTX 1050 Ti | sm_61 | เฉลี่ย ~0.50–0.53 MH/s | 2026-11-26, เก็บ convolution factor เป็น int8 พร้อม fallback กลับไป int16 อัตโนมัติ รัน 170 วิสองรอบกับ pool ท้องถิ่น `neb-pool/1` และชุดข้อมูลทางการได้เฉลี่ย 529.02 และ 506.61 kH/s ขณะที่รอบ baseline int16 ที่อยู่ข้างกันได้ 517.30 และ 486.20 kH/s share ทุกตัวถูกยอมรับ ไม่มี reject |

การลด shared memory ของ linear layer จาก `4 × (cout + cin)` เป็น `4 × cout`
words ทำให้ ticket เหมือนเดิมทุกตัวในการทดสอบ differential แต่ความเร็วไม่
เปลี่ยนมากนัก การเปรียบเทียบ A/B ระหว่าง warp reduction กับ shared-atomic
แบบเดิมพบว่าเร็วขึ้นราว 3% การ precompute row offset แบบ 32 บิต, การใช้
`__ldg` กับแถวของตาราง และการลด `THREADS` จาก 256 เป็น 128 ทำให้เฉลี่ย 5 นาที
เพิ่มจาก ~0.34 เป็น ~0.41–0.43 MH/s ตอนนี้เก็บ convolution factor เป็น int8
เมื่อ gradient ทุกตัวของทุกแถวใส่ได้ และ builder จะลอง layout int16 เดิมทันทีถ้ามี
ค่าเกินช่วง จึงทำให้ ticket เหมือนเดิมทุกตัว การ transpose สี่แถวเป็นคำเดียวแล้วรวมด้วย
`__dp4a` ช่วยเพิ่มอีกประมาณ 2–4% บน GTX 1050 Ti ตัวแปรที่ผ่าน differential แต่ถูก
ทิ้งเพราะช้าหรือเสถียรน้อยกว่าได้แก่ staging แบบ transposed 32 บิต, การโหลด
convolution weight สองตัวต่อครั้ง, `dp4a` แบบไม่มี branch, row offset ที่ precompute
ใน shared memory, 512 threads และ 64 threads ส่วนตัวแปร `__dp2a` สำหรับ int16
ถูกทิ้งทันทีเพราะ differential ไม่ผ่าน

### ผลทดสอบ kernel รุ่นรวมบน Yiimp จริง

ทดสอบแต่ละรุ่นบน GTX 1050 Ti (`sm_61`) รอบละ 90 วินาที ใช้ชุดข้อมูลทางการ
target ของแชร์ตรงกันทั้ง 32 ไบต์ (`d=0.0002`) และ batch ละ 8,192 nonce
อัตราสแกนคิดจากเวลาที่ใช้ใน CUDA scan call ส่วนเวลามัธยฐานต่อ batch
วัดจากตอนเรียกจนได้ผลกลับมา

| วิธี | อัตราขณะสแกน | เวลามัธยฐานต่อ batch | แชร์ที่พูลรับ |
| :-- | --: | --: | --: |
| BLAKE3 cache อย่างเดียว | 577.97 kH/s | 14.32 ms | 8/8 |
| ข้าม gradient ศูนย์ใน linear layer อย่างเดียว | 647.91 kH/s | 12.42 ms | 22/22 |
| รวมทั้งสองวิธี | 651.25 kH/s | 12.16 ms | 7/8 |

รุ่นรวมสแกนจริง 18.5 วินาที ส่วนรุ่นข้าม gradient ศูนย์อย่างเดียวสแกนจริง
57.4 วินาที Yiimp ปฏิเสธหนึ่งแชร์ของรุ่นรวมด้วยรหัส 21
(`Job not found or past its expiry`) ระหว่างเปลี่ยนงาน ความเร็วที่เพิ่มขึ้น
ส่วนใหญ่มาจากการข้าม gradient ศูนย์ ส่วนต่าง 0.5% ระหว่างรุ่นรวมกับรุ่นข้าม
gradient ศูนย์อย่างเดียวยังต้องทดสอบซ้ำเพื่อยืนยันผล

### ผลสะสม linear ต่อ thread บน Yiimp จริง

รุ่นใหม่สแกนได้ 741.42 kH/s บน GTX 1050 Ti (`sm_61`) เพิ่มขึ้น 17.6%
จาก `15294b5` ทั้งสองรุ่นใช้ชุดข้อมูลทางการ batch ละ 8,192 nonce และ
target ของแชร์ตรงกันครบ 32 ไบต์ (`d=0.0002`) ทดสอบตามลำดับรุ่นอ้างอิง
รุ่นใหม่ รุ่นใหม่ และรุ่นอ้างอิง แต่ละรอบจบเมื่อ Yiimp ประกาศความสูงที่ต่างกัน
ครบสามค่า โดยนับการแจ้งงานซ้ำที่ความสูงเดิมเพียงครั้งเดียว

| Kernel | อัตราขณะสแกน | เวลามัธยฐานต่อ batch | ส่ง | พูลรับ | ไม่มีคำตอบ |
| :-- | --: | --: | --: | --: | --: |
| รุ่นอ้างอิง `15294b5` | 630.68 kH/s | 12.78 ms | 59 | 31 | 28 |
| สะสมผล linear ต่อ thread | 741.42 kH/s | 10.68 ms | 33 | 33 | 0 |

รุ่นใหม่สแกนได้ 737.67 และ 743.24 kH/s ในสองรอบ การเชื่อมต่อถูกรีเซ็ตใน
รอบของรุ่นอ้างอิง และมี 28 รายการที่ไม่มีคำตอบใน log จึงมีข้อมูลการตอบรับ
แชร์ครอบคลุมต่างกันระหว่างสองรุ่น รุ่นใหม่ผ่านการเทียบกับ C reference ด้วย
384 nonce สำหรับทั้ง KEP1 และ RET2 ครบห้ารูปแบบน้ำหนักต่อสถาปัตยกรรม
ห้าขอบเขต rank และแปด ticket แบบตรงค่าต่อกรณี
