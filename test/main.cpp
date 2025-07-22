#include <iostream>
#include "../fun/Matriz2D.h"
#include "../fun/CapaTokenizacion.h"
int main() {
    std::cout << "=== Prueba de Matriz2D con CUDA ===\n";
    srand(time(NULL));

    // Crear dos matrices aleatorias
    Matriz2D A(3, 3, 1); // 3x3 con valores aleatorios
    Matriz2D B(3, 3, 1); // 3x3 con valores aleatorios

    std::cout << "Matriz A:\n" << A << "\n";
    std::cout << "Matriz B:\n" << B << "\n";

    // ✅ Multiplicación con CUDA
    //Matriz2D C = A * B;
    Matriz2D C = A.MultiplicarCUDA(B);
    std::cout << "Resultado A * B (CUDA):\n" << C << "\n";

    // ✅ Aplicar ReLU con CUDA
    std::cout << "Aplicando ReLU (CUDA) a A...\n";
    A.RELU_CUDA();
    std::cout << "A después de ReLU:\n" << A << "\n";

    // ✅ Softmax por filas con CUDA
    std::cout << "Aplicando Softmax (CUDA) a B...\n";
    B.SoftmaxFilas_CUDA();
    std::cout << "B después de Softmax:\n" << B << "\n";

    // ✅ Normalización por filas con CUDA
    std::cout << "Aplicando Normalización (CUDA) a C...\n";
    C.NormalizarFilas_CUDA();
    std::cout << "C después de Normalizar:\n" << C << "\n";

    
    
    // Crear capa con dimensión de embedding 8
    CapaTokenizacion capa(8);

    std::string texto = "hola mundo";
    Matriz2D salida;

    capa.Forward(texto, salida);

    std::cout << "\nMatriz tokenizada (embedding):\n";
    //salida.Imprimir();  // o sobrecarga de operador <<
    std::cout << "Matriz tokenizada (embedding):\n" << salida << "\n";

    return 0;
}
