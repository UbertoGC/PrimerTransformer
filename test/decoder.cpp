#include <iostream>
#include "../fun/Decoder.h"
#include <chrono>

using namespace std;

int main() {
    // Texto de prueba
    string PruebaTexto = "I'll have 20";
    PruebaTexto.push_back('%');
    PruebaTexto += " of it.";

    string PruebaTexto2 = "GPT2 was created by OpenAI and I am using it";

    // Crear decoder
    Decoder GPT1;

    // Bandera: Cambia a true para usar CUDA
    bool usarCUDA = true;  // (true = GPU, false = CPU)

    cout << "=== Ejecutando Decoder ===" << endl;
    cout << "Modo: " << (usarCUDA ? "GPU (CUDA)" : "CPU") << endl;

    // Medir tiempo
    auto inicio = chrono::high_resolution_clock::now();

    GPT1.Ejecutar(PruebaTexto2, usarCUDA);

    auto fin = chrono::high_resolution_clock::now();
    chrono::duration<double> duracion = fin - inicio;

    cout << "\nTiempo de ejecución: " << duracion.count() << " segundos" << endl;

    return 0;
}
