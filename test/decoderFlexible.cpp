#include <iostream>
#include "../fun/DecoderFlexible.h"
#include <chrono>
#include <iomanip>

// Función para ejecutar y medir tiempos
void benchmarkEjecucion(DecoderFlexible& decoder,  std::string& texto, bool usarCUDA) {
    Matriz2D salida;
    
    auto inicio = std::chrono::high_resolution_clock::now();
    
    decoder.Ejecutar(texto, salida, usarCUDA);
    
    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duracion = fin - inicio;
    
    std::cout << "\n  Tiempo " << (usarCUDA ? "GPU (CUDA)" : "CPU") 
              << ": " << std::fixed << std::setprecision(4) 
              << duracion.count() << " segundos\n";
    
    // Calcular tokens/segundo
    size_t num_tokens = salida.fil();
    double tokens_por_segundo = num_tokens / duracion.count();
    
    std::cout << " Rendimiento: " << std::setprecision(2) 
              << tokens_por_segundo << " tokens/segundo\n";
}

int main() {
    // Configuración
    const int num_pruebas = 3;
    std::string textos[num_pruebas] = {
        "Transformers son modelos poderosos",
        "La inteligencia artificial está revolucionando el mundo",
        "GPT es un modelo de lenguaje avanzado"
    };
    
    DecoderFlexible decoder(2); // 2 bloques
    
    std::cout << "=== BENCHMARK DECODER (CPU vs GPU) ===\n";
    std::cout << "Pruebas con textos de diferente longitud:\n";
    
    for (int i = 0; i < num_pruebas; ++i) {
        std::cout << "\n Prueba " << i+1 << "/" << num_pruebas 
                  << " - Longitud: " << textos[i].length() << " caracteres\n";
        
        // Ejecutar en GPU
        std::cout << "\n[GPU] ";
        benchmarkEjecucion(decoder, textos[i], true);
        
        // Ejecutar en CPU
        std::cout << "[CPU] ";
        benchmarkEjecucion(decoder, textos[i], false);
        
        // Calcular speedup
        Matriz2D dummy;
        auto inicio_gpu = std::chrono::high_resolution_clock::now();
        decoder.Ejecutar(textos[i], dummy, true);
        auto fin_gpu = std::chrono::high_resolution_clock::now();
        
        auto inicio_cpu = std::chrono::high_resolution_clock::now();
        decoder.Ejecutar(textos[i], dummy, false);
        auto fin_cpu = std::chrono::high_resolution_clock::now();
        
        double tiempo_gpu = std::chrono::duration<double>(fin_gpu - inicio_gpu).count();
        double tiempo_cpu = std::chrono::duration<double>(fin_cpu - inicio_cpu).count();
        double speedup = tiempo_cpu / tiempo_gpu;
        
        std::cout << " Speedup GPU vs CPU: " << std::setprecision(2) 
                  << speedup << "x\n";
        
        std::cout << "----------------------------------------\n";
    }
    
    return 0;
}