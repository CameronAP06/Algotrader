@echo off
:: ─── ROCm / MIOpen workarounds for gfx1100 (RX 7000) + MSVC 14.39 ───────────
:: Force HIP's JIT compiler to use MSVC 14.38 toolset instead of 14.39.
:: The 14.39 toolset added [[nodiscard]] to std::forward which conflicts with
:: MIOpen's CK library miopen_utility.hpp — causing gridwise_generic_reduction
:: kernel compilation to fail with "redefinition of 'forward'".
set VCToolsVersion=14.34.31933
set VCToolsInstallDir=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.34.31933\

:: MIOpen tuning — suppress unnecessary kernel search/logging
set MIOPEN_FIND_ENFORCE=NONE
set MIOPEN_DEBUG_DISABLE_FIND_DB=1
set MIOPEN_LOG_LEVEL=0

:: ─── Launch ──────────────────────────────────────────────────────────────────
echo VCToolsVersion: %VCToolsVersion%
echo Starting timeframe comparison...
echo Timeframes: 15m 1h 2h 4h 8h 1d
echo.

python timeframe_comparison.py --walkforward --timeframes 15m 1h 2h 4h 8h 1d --refresh

echo.
echo Training complete.
pause