# compilar.ps1

Write-Host "==== Compilando con NVCC ====" -ForegroundColor Cyan

# Compilar el proyecto
nvcc test/main.cpp fun/Matriz_Cuda.cu -I./fun -o ejecutable -std=c++11 -Wno-deprecated-gpu-targets

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Compilación exitosa. Ejecutando el programa..." -ForegroundColor Green
    ./ejecutable
} else {
    Write-Host "`n❌ Error en la compilación. Corrige los errores." -ForegroundColor Red
}
