#include <iostream>
#include <fstream>
#include <vector>
#include "../fun/DecoderFlexible.h"
#include <chrono>
#include <iomanip>

struct BenchmarkResult {
    int d_model;         // Nuevo: dimensión del modelo
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
    archivo << "d_model,Texto,Longitud,Tiempo_GPU(ms),Tiempo_CPU(ms),Speedup,Tokens/s_GPU,Tokens/s_CPU\n";
    for (const auto& res : resultados) {
        archivo << res.d_model << ","
                << "\"" << res.texto << "\","
                << res.longitud << ","
                << res.tiempo_gpu * 1000 << ","
                << res.tiempo_cpu * 1000 << ","
                << res.speedup << ","
                << res.tokens_por_segundo_gpu << ","
                << res.tokens_por_segundo_cpu << "\n";
    }
}

void benchmarkEjecucion(DecoderFlexible& decoder,  std::string& texto, bool usarCUDA, 
                       double& tiempo, double& tokens_por_segundo) {
    Matriz2D salida;
    auto inicio = std::chrono::high_resolution_clock::now();
    decoder.Ejecutar(texto, salida, usarCUDA);
    tiempo = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - inicio).count();
    tokens_por_segundo = salida.fil() / tiempo;
}

int main() {
    // Configuración de pruebas
    const std::vector<int> d_models = {32, 64, 128, 256, 512, 1024,2048,4096};  // Valores a probar
     std::vector<std::string> textos = {
        "Transformers son modelos poderosos",
        "La inteligencia artificial está revolucionando el mundo"
    };
    std::vector<BenchmarkResult> resultados;

    std::cout << "=== BENCHMARK COMPARATIVO (d_model) ===\n";
    
    for (int d_model : d_models) {
        std::cout << "\n🔧 Configurando d_model = " << d_model << "\n";
        DecoderFlexible decoder(2, 4, 128,d_model); // 2 bloques, 4 cabezas, d_ff=128,d_model=d_model
        
        // Warm-up
        Matriz2D dummy;
        std::string warmup_text = "Warm-up for d_model " + std::to_string(d_model);
        decoder.Ejecutar(warmup_text, dummy, true);

        for ( auto& texto : textos) {
            BenchmarkResult res;
            res.d_model = d_model;
            res.texto = texto;
            res.longitud = texto.length();

            // GPU
            benchmarkEjecucion(decoder, texto, true, res.tiempo_gpu, res.tokens_por_segundo_gpu);
            // CPU
            benchmarkEjecucion(decoder, texto, false, res.tiempo_cpu, res.tokens_por_segundo_cpu);
            
            res.speedup = res.tiempo_cpu / res.tiempo_gpu;
            resultados.push_back(res);

            std::cout << "  Longitud " << res.longitud << " | GPU: " 
                      << res.tiempo_gpu * 1000 << " ms | CPU: " 
                      << res.tiempo_cpu * 1000 << " ms | Speedup: " 
                      << res.speedup << "x\n";
        }
    }

    guardarResultadosCSV(resultados, "benchmark_d_model.csv");
    std::cout << "\n✅ Resultados guardados en benchmark_d_model.csv\n";

    return 0;
}