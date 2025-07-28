#include <iostream>
#include "../fun/Matriz2D.h"
#include "../fun/CapaTokenizacion.h"
#include "../fun/CapaEmbedding.h"
#include "../fun/CapaFeedForward.h"
#include "../fun/CapaAtencion.h"
#include "../fun/BloqueTransformer.h"

int main() {
    std::cout << "=== Prueba de Matriz2D con CUDA (modo Decoder) ===\n";
    srand(time(NULL));

    // ==============================
    // 1. Parámetros grandes
    // ==============================
    int d_model = 768;
    int num_cabezas = 12;
    int d_cabeza = d_model / num_cabezas;
    int d_ff = 3072;
    int tam_seq = 16; // evitar explotar memoria
    int vocab_size = 50257;

    // ==============================
    // 2. Crear entrada simulada
    // ==============================
    Matriz2D tokens(tam_seq, d_model);
    tokens.Random();

    std::cout << "Matriz simulada de tokens [" << tokens.fil() << " x " << tokens.col() << "]\n";

    // ==============================
    // 3. Capa Embedding
    // ==============================
    std::cout << "\n=== Capa Embedding ===\n";
    CapaEmbedding capaEmb(tam_seq, d_model);

    Matriz2D salidaEmbedding;
    capaEmb.Forward(tokens, salidaEmbedding, true);
    std::cout << "Salida Embedding: [" << salidaEmbedding.fil() << " x " << salidaEmbedding.col() << "]\n";

    // ==============================
    // 4. Capa Atención (MHA)
    // ==============================
    std::cout << "\n=== Capa Atencion ===\n";
    CapaAtencion capaAtencion(num_cabezas, d_model, d_cabeza);

    Matriz2D salidaAtencion;
    capaAtencion.Forward(salidaEmbedding, salidaAtencion, true);
    std::cout << "Salida Atencion: [" << salidaAtencion.fil() << " x " << salidaAtencion.col() << "]\n";

    // ==============================
    // 5. FeedForward
    // ==============================
    std::cout << "\n=== FeedForward ===\n";
    CapaFeedForward capaFF(d_ff, d_model);

    Matriz2D salidaFF;
    capaFF.Forward(salidaAtencion, salidaFF, true);
    std::cout << "Salida FeedForward: [" << salidaFF.fil() << " x " << salidaFF.col() << "]\n";

    // ==============================
    // 6. Bloque Transformer
    // ==============================
    std::cout << "\n=== Bloque Transformer ===\n";
    BloqueTransformer bloque(d_ff, num_cabezas, d_model, d_cabeza, tam_seq);

    Matriz2D salidaTransformer;
    bloque.Forward(salidaEmbedding, salidaTransformer, true);
    std::cout << "Salida Transformer: [" << salidaTransformer.fil() << " x " << salidaTransformer.col() << "]\n";

    return 0;
}
