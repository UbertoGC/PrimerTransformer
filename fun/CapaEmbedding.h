#ifndef __CAPAEMBEDDING_H__
#define __CAPAEMBEDDING_H__

#include <iostream>
#include "Matriz2D.h"

class CapaEmbedding {
private:
    Matriz2D posicion_embedding;
    int d_modelo;

public:
    CapaEmbedding(int max_entrada, int d_m);
    void Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA = false);
    ~CapaEmbedding();
     int GetMaxLen() const { return posicion_embedding.fil(); }
};

CapaEmbedding::CapaEmbedding(int max_entrada, int d_m) {
    d_modelo = d_m;
    posicion_embedding.ReSize(max_entrada, d_modelo);
    posicion_embedding.Random();
    std::cout << "Capa de Embedding creada (max=" << max_entrada
              << ", d_model=" << d_modelo << ")" << std::endl;
}

void CapaEmbedding::Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA) {
    salida.ReSize(entrada.fil(), d_modelo);
    std::cout << "Aplicando Embedding Posicional..." << std::endl;
    if (usarCUDA) {
        //  Copiar embeddings y aplicar suma en GPU
        std::cout<<"entra sumacuda"<<std::endl;
        salida.SumarMatrizCUDA(posicion_embedding);
        std::cout << "Embedding aplicado en GPU." << std::endl;
    } else {
        //  Versión CPU
        for (int i = 0; i < entrada.fil(); i++) {
            for (int j = 0; j < d_modelo; j++) {
                salida(i, j) = posicion_embedding(i, j);
            }
        }
    }
}

CapaEmbedding::~CapaEmbedding() {}

#endif
