#ifndef __DECODER_FLEXIBLE_H__
#define __DECODER_FLEXIBLE_H__

#include "Matriz2D.h"
#include "CapaTokenizacion.h"
#include "CapaEmbedding.h"
#include "BloqueTransformer.h"
#include <vector>

class DecoderFlexible {
private:
    CapaTokenizacion* tokenizador;
    std::vector<BloqueTransformer*> bloques;
    int d_model;
    int num_cabezas;
    int d_ff;

    void LogDatos(const Matriz2D& mat, const std::string& nombre) {
        std::cout << nombre << " size: [" << mat.fil() 
                  << " x " << mat.col() << "]\n";
        #ifdef DEBUG_DETAIL
        std::cout << "Contenido:\n" << mat << "\n";
        #endif
    }

public:
    DecoderFlexible(int num_bloques = 2, int num_cabezas = 4, int d_ff = 128,int d_model = 32);
    void Ejecutar(std::string& texto, Matriz2D& salida, bool usarCUDA = false);
    ~DecoderFlexible();
};

// Implementación
DecoderFlexible::DecoderFlexible(int num_bloques, int num_cabezas, int d_ff, int d_model) 
    : num_cabezas(num_cabezas), d_ff(d_ff), d_model(d_model) {
    
    std::cout << "=== Creando DecoderFlexible ===\n";
    tokenizador = new CapaTokenizacion(d_model); // Tamaño de vocabulario arbitrario
    bloques.resize(num_bloques, nullptr);
    std::cout << "Configurado para " << num_bloques << " bloques\n";
}

void DecoderFlexible::Ejecutar(std::string& texto, Matriz2D& salida, bool usarCUDA) {
    std::cout << "\n=== Ejecutando Decoder ===\n";
    std::cout << "Modo: " << (usarCUDA ? "GPU (CUDA)" : "CPU") << "\n";
    std::cout << "Texto entrada: \"" << texto << "\"\n";

    // 1. Tokenización
    Matriz2D tokens;
    tokenizador->Forward(texto, tokens);
    d_model = tokens.col();
    const int seq_len = tokens.fil();
    LogDatos(tokens, "Tokens");

    // 2. Embedding
    std::cout << "\n[1/4] Aplicando Embedding Posicional...\n";
    CapaEmbedding embedding(seq_len, d_model);
    Matriz2D emb_output;
    embedding.Forward(tokens, emb_output, usarCUDA);
    LogDatos(emb_output, "Embedding");

    // 3. Bloques Transformer
    Matriz2D bloque_input = emb_output;
    for (size_t i = 0; i < bloques.size(); ++i) {
        if (!bloques[i]) {
            bloques[i] = new BloqueTransformer(
                d_ff, num_cabezas, d_model, 
                d_model/num_cabezas, seq_len
            );
            std::cout << "Inicializado bloque " << i+1 << "\n";
        }

        std::cout << "\n[2." << i+1 << "/4] Procesando Bloque " << i+1 << "...\n";
        Matriz2D bloque_output;
        bloques[i]->Forward(bloque_input, bloque_output, usarCUDA);
        LogDatos(bloque_output, "Salida bloque");

        bloque_input = bloque_output;
        if (i == bloques.size() - 1) {
            salida = bloque_output;
        }
    }

    std::cout << "\n=== Resultado Final ===\n";
    LogDatos(salida, "Salida Decoder");
}

DecoderFlexible::~DecoderFlexible() {
    delete tokenizador;
    for (auto bloque : bloques) {
        delete bloque;
    }
    std::cout << "DecoderFlexible liberado\n";
}

#endif