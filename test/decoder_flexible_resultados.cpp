#include <iostream>
#include <fstream>
#include <vector>
#include "../fun/DecoderFlexible.h"
#include <chrono>
#include <iomanip>

struct BenchmarkResult {
    std::string texto;
    int longitud;
    double tiempo_gpu;
    double tiempo_cpu;
    double speedup;
    double tokens_por_segundo_gpu;
    double tokens_por_segundo_cpu;
};

void guardarResultadosCSV(const std::vector<BenchmarkResult>& resultados, const std::string& filename) {
    std::ofstream archivo(filename);
    if (!archivo.is_open()) {
        std::cerr << "Error al abrir el archivo " << filename << std::endl;
        return;
    }

    // Encabezados CSV
    archivo << "Texto,Longitud,Tiempo_GPU(ms),Tiempo_CPU(ms),Speedup,Tokens_por_segundo_GPU,Tokens_por_segundo_CPU\n";

    // Datos
    for (const auto& res : resultados) {
        archivo << "\"" << res.texto << "\","
                << res.longitud << ","
                << res.tiempo_gpu * 1000 << ","  // Convertir a milisegundos
                << res.tiempo_cpu * 1000 << ","
                << res.speedup << ","
                << res.tokens_por_segundo_gpu << ","
                << res.tokens_por_segundo_cpu << "\n";
    }

    std::cout << "\nResultados guardados en " << filename << std::endl;
}

void benchmarkEjecucion(DecoderFlexible& decoder,  std::string& texto, bool usarCUDA, 
                       double& tiempo, double& tokens_por_segundo) {
    Matriz2D salida;
    auto inicio = std::chrono::high_resolution_clock::now();
    
    decoder.Ejecutar(texto, salida, usarCUDA);
    
    auto fin = std::chrono::high_resolution_clock::now();
    tiempo = std::chrono::duration<double>(fin - inicio).count();
    tokens_por_segundo = salida.fil() / tiempo;
}

int main() {
    // Configuración
    std::vector<std::string> textos = {
        "Transformers son modelos poderosos",
        "La inteligencia artificial está revolucionando el mundo",
        "GPT es un modelo de lenguaje avanzado",
        "La atención multi-cabeza es clave en los transformers",
        "CUDA acelera el procesamiento de modelos profundos"
    };

    DecoderFlexible decoder(2); // 2 bloques
    std::vector<BenchmarkResult> resultados;

    std::cout << "=== BENCHMARK DECODER (CPU vs GPU) ===\n";
    std::cout << "Pruebas con " << textos.size() << " textos diferentes\n";

    // Warm-up GPU (opcional pero recomendado)
    {
        std::string warmup_text = "Warm-up inicial";
        Matriz2D dummy;
        decoder.Ejecutar(warmup_text, dummy, true);
    }

    for (auto& texto : textos) {
        BenchmarkResult res;
        res.texto = texto;
        res.longitud = texto.length();

        std::cout << "\n🔍 Prueba - Longitud: " << res.longitud << " caracteres\n";

        // Benchmark GPU
        std::cout << "[GPU] ";
        benchmarkEjecucion(decoder, texto, true, res.tiempo_gpu, res.tokens_por_segundo_gpu);
        std::cout << "  Tiempo GPU: " << std::fixed << std::setprecision(2) 
                  << res.tiempo_gpu * 1000 << " ms | "
                  << res.tokens_por_segundo_gpu << " tokens/s\n";

        // Benchmark CPU
        std::cout << "[CPU] ";
        benchmarkEjecucion(decoder, texto, false, res.tiempo_cpu, res.tokens_por_segundo_cpu);
        std::cout << "  Tiempo CPU: " << std::fixed << std::setprecision(2) 
                  << res.tiempo_cpu * 1000 << " ms | "
                  << res.tokens_por_segundo_cpu << " tokens/s\n";

        // Calcular speedup
        res.speedup = res.tiempo_cpu / res.tiempo_gpu;
        std::cout << "⚡ Speedup GPU vs CPU: " << std::setprecision(2) 
                  << res.speedup << "x\n";

        resultados.push_back(res);
    }

    // Guardar resultados
    guardarResultadosCSV(resultados, "benchmark_results.csv");

    return 0;
}