compile ccminer.exe for windows 10, 11
# Instructions for Compiling CCMiner on Windows 10/11

## I. SYSTEM REQUIREMENTS

### Hardware:
- NVIDIA GPU (CUDA support)
- RAM: minimum 8GB (16GB recommended)
- Hard drive: ~20GB free space

### Software:

1. **Visual Studio 2022** (Community Edition - free)

- Download: https://visualstudio.microsoft.com/downloads/

2. **NVIDIA CUDA Toolkit** (choose 1 of 3 versions):

- CUDA 10.2: https://developer.nvidia.com/cuda-10.2-download-archive
- CUDA 11.7: https://developer.nvidia.com/cuda-11-7-0-download-archive
- CUDA 12.8: https://developer.nvidia.com/cuda-downloads (recommended for new GPUs)

3. **Git for Windows** (to clone source code)
- Download: https://git-scm.com/download/win

---

## II. INSTALL TOOLS

### Step 1: Install Visual Studio 2022

1. Run the Visual Studio setup file
2. Select workload: **"Desktop development with C++"**
3. In the "Individual components" tab, make sure to select:
- MSVC v143 - VS 2022 C++ x64/x86 build tools
- Windows 10 SDK or Windows 11 SDK
- C++ CMake tools for Windows
4. Click Install and wait for completion

### Step 2: Install CUDA Toolkit

1. Download the CUDA Toolkit that matches your GPU:
- GTX 10xx GPU or earlier: CUDA 10.2
- RTX 20xx, 30xx GPU: CUDA 11.7
- RTX 40xx GPU or later: CUDA 12.8

2. Run the setup file, select **"Custom"** installation
3. Select the component:
- CUDA Toolkit
- Visual Studio Integration
- Nsight Systems
4. Install to the default directory: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

### Step 3: Check the installation

Open **Command Prompt** and check:

```cmd
nvcc --version
```

The result will display the installed CUDA version.

---

## III. DOWNLOAD AND UNZIP THE SOURCE CODE

### Method 1: Using Git (recommended)

```cmd
cd C:\
git clone https://github.com/tpruvot/ccminer.git
cd ccminer
```

### Method 2: Download ZIP

1. Unzip the source code to the folder: `C:\ccminer`
2. Make sure the folder structure is correct with the .sln and .vcxproj files

---

## IV. COMPILING CCMINER

### Method 1: Using Visual Studio (Recommended for beginners)

1. **Open Project:**
- Select the solution file that matches the installed CUDA:
- `ccminer.sln` (CUDA 12.8)
- `ccminer-cuda11.sln` (CUDA 11.7)
- `ccminer-cuda10.sln` (CUDA 10.2)
- Double-click to open with Visual Studio 2022

2. **Select Configuration:**
- On the toolbar, select:
- Configuration: **Release**
- Platform: **x64** (64-bit, recommended)

3. **Build Project:**
- Menu: `Build` → `Build Solution` (or Ctrl+Shift+B)
- Wait for the compilation process (may take 10-30 minutes)

4. **Get the .exe file:**

- The ccminer.exe file will be in:

- `x64\Release\ccminer.exe`

### Method 2: Using Command Line (Build script)

1. **Open Developer Command Prompt:**
- Search: "Developer Command Prompt for VS 2022" in Start Menu
- Run as Administrator

2. **Navigate to the source folder:**
```cmd
cd C:\ccminer
```

3. **Build with MSBuild:**
```cmd
msbuild ccminer.vcxproj /p:Configuration=Release /p:Platform=x64 /m
```

Or use the available script:
```cmd
build.cmd
```

---

## V. TROUBLESHOOTING COMMON ERRORS ENCOUNTER

### Error 1: "CUDA Toolkit not found"

**Solution:**
1. Check the environment variable `CUDA_PATH`:
```cmd
echo %CUDA_PATH%
```

2. If empty, add it manually:
- Control Panel → System → Advanced → Environment Variables
- Add variable: `CUDA_PATH` = `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

### Error 2: "MSB8036: The Windows SDK version was not found"

**Solution:**
- Open the file `.vcxproj` with Notepad
- Find the line: `<WindowsTargetPlatformVersion>`
- Change it to: `<WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>`

### Error 3: "error C2039: 'CapsuleGeometry' is not a member of 'THREE'"

**Solution:**
- This error with old CUDA, Three.js library is not compatible
- Ignore or use CUDA 11.7+ with the corresponding project

### Error 4: "LNK1181: cannot open input file 'cudart_static.lib'"

**Solution:**
1. Add CUDA lib path to Project Properties:
- Right-click project → Properties
- Configuration Properties → Linker → General
- Additional Library Directories: `$(CUDA_PATH)\lib\x64`

### Error 5: "Out of memory" during build

**Solution:**
- Reduce the number of build threads: Remove `/m` in msbuild command
- Close other applications to free up RAM

---

## VI. CHECK AND RUN

### Check the exe file:

```cmd
cd x64\Release
ccminer.exe --version
```

The result shows the version and build information.

### Test mining (example):

```cmd
ccminer.exe -a allium -o stratum+tcp://pool.example.com:3333 -u wallet.worker -p x -n
```

The `-n` parameter is used to test without connecting to the pool.

---

## VII. BUILD OPTIMIZATION

### Increase build speed:

1. **Enable Multi-processor Compilation:** 
- Project Properties → C/C++ → General 
- Multi-processor Compilation: **Yes (/MP)**

2. **Disable Debug Info in Release:** 
- Project Properties → Linker → Debugging 
- Generate Debug Info: **No**

### Build for specific GPU:

Open file `.vcxproj`, find section `<CodeGeneration>` and only retain the compute capability of the GPU:

**Example for RTX 3060 (compute_86):**
```xml
<CodeGeneration>
compute_86,sm_86;
</CodeGeneration>
```
**Compute Capabilities:**
- GTX 1080: compute_61
- RTX 2080: compute_75
- RTX 3060/3070/3080: compute_86
- RTX 4090: compute_89

---

## VIII. INCLUDED DEPENDENCIES

The `ccminer.exe` file requires the following DLLs (usually copied automatically):

- `pthreadVC2.dll` (in the `compat\pthreads\x64` folder)
- `cudart64_XX.dll` (from CUDA Toolkit, XX is the version)

Copy these files to the same folder as ccminer.exe if it reports a missing DLL error.

---

## IX. IMPORTANT NOTES

1. **Antivirus:** Turn off antivirus when building and running, as miner is often mistakenly recognized as malware

2. **Driver:** Update the latest NVIDIA driver from: https://www.nvidia.com/Download/index.aspx

3. **Compute Capability:** Only build for your GPU to reduce exe file size

4. **Admin Rights:** Some algorithms require admin rights to run

5. **Pool Testing:** Test pool connection before actual mining

---

## X. CONCLUSION

Once completed, you will have a `ccminer.exe` file that can run independently on Windows 10/11. This file can:

- Mining coins using supported algorithms
- Benchmark GPU performance
- Connect to mining pools

**References:**
- README.txt in the source code for supported algorithms
- Command line options: `ccminer.exe --help`

Happy building! 🚀
