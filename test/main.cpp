#include <iostream>
#include "../fun/Matriz2D.h"
#include "../fun/CapaTokenizacion.h"
#include "../fun/CapaEmbedding.h"
#include "../fun/CapaFeedForward.h"
#include "../fun/CapaAtencion.h"
#include "../fun/BloqueTransformer.h"
int main() {
    std::cout << "=== Prueba de Matriz2D con CUDA ===\n";
    srand(time(NULL));

    // ==============================
    // 1. Pruebas básicas con Matriz2D
    // ==============================
    Matriz2D A(3, 3, 1); // 3x3 aleatoria
    Matriz2D B(3, 3, 1); // 3x3 aleatoria

    std::cout << "Matriz A:\n" << A << "\n";
    std::cout << "Matriz B:\n" << B << "\n";

    // ✅ Multiplicación con CUDA
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
    // Crear gamma y beta
    Matriz2D gamma(1, C.col());
    Matriz2D beta(1, C.col());
    gamma.Fill(1.0f); // γ = 1
    beta.Fill(0.0f);  // β = 0
    C.NormalizarFilas_CUDA(gamma, beta);
    std::cout << "C después de Normalizar:\n" << C << "\n";

    // ==============================
    // 2. Tokenización
    // ==============================
    std::cout << "\n=== Tokenización ===\n";
    CapaTokenizacion capaToken(8);  // embedding inicial de tamaño 8
    std::string texto = "hola mundo";
    Matriz2D tokens;
    capaToken.Forward(texto, tokens);

    std::cout << "Matriz tokenizada (embedding inicial):\n" << tokens << "\n";

    // ==============================
    // 3. Capa Embedding (Positional Encoding)
    // ==============================
    std::cout << "\n=== Capa Embedding ===\n";
    int max_len = tokens.fil();
    int d_model = tokens.col();
    CapaEmbedding capaEmb(max_len, d_model);

    Matriz2D salidaEmbedding;
    capaEmb.Forward(tokens, salidaEmbedding, true);
    std::cout << "Salida después de Embedding (con posiciones):\n" << salidaEmbedding << "\n";

    // ==============================
    // 4. Capa Atención (Multi-Head Attention)
    // ==============================
    std::cout << "\n=== Capa Atencion ===\n";
    int num_cabezas = 2;      // por ejemplo, 2 cabezas
    int d_cabeza = d_model / num_cabezas; // tamaño por cabeza
    CapaAtencion capaAtencion(num_cabezas, d_model, d_cabeza);

    Matriz2D salidaAtencion;
    capaAtencion.Forward(salidaEmbedding, salidaAtencion,true);
    std::cout << "Salida después de Atencion:\n" << salidaAtencion << "\n";

    // ==============================
    // 5. Capa FeedForward (GPU)
    // ==============================
    std::cout << "\n=== Capa FeedForward ===\n";
    int d_ff = 16; // dimensión interna
    CapaFeedForward feedForward(d_ff, d_model);

    Matriz2D salidaFF;
    feedForward.Forward(salidaAtencion, salidaFF, true); // true = usar CUDA
    std::cout << "Salida después de FeedForward:\n" << salidaFF << "\n";

    // ==============================
    // 6. Probar BloqueTransformer
    // ==============================
    // ==============================
    // 6. Probar BloqueTransformer
    // ==============================
    std::cout << "\n=== Bloque Transformer ===\n";
    //  Eliminar la redeclaración, usar las existentes
    int tam_seq = salidaEmbedding.fil();

    BloqueTransformer bloque(d_ff, num_cabezas, d_model, d_cabeza, tam_seq);

    Matriz2D salidaTransformer;
    bloque.Forward(salidaEmbedding, salidaTransformer, true); // usar CUDA
    std::cout << "Salida después de BloqueTransformer:\n" << salidaTransformer << "\n";


    return 0;
}
