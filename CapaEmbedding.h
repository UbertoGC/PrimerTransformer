#ifndef __CAPAEMBEDDING_H__
#define __CAPAEMBEDDING_H__

#include <vector>
#include <string>
#include "Matriz2D.h"

class CapaEmbedding
{
private:
    Matriz2D token_embedding;
    Matriz2D posicion_embedding;
    Matriz2D* embedding_matriz;
    int vocabulario_size;
    int d_modelo;
public:
    CapaEmbedding(int, int);
    void Forward(std::vector<int>&, Matriz2D&);
    ~CapaEmbedding();
};
CapaEmbedding::CapaEmbedding(int vocab_size, int d_model){
    vocabulario_size = vocab_size;
    d_modelo = d_model;
    embedding_matriz = nullptr;
    posicion_embedding.ReSize(512, d_modelo);
    posicion_embedding.Random();
    std::cout<<"Posicion Embedding size: ["<<posicion_embedding.fil()<<" x "<<posicion_embedding.col()<<"]"<<std::endl;
    token_embedding.ReSize(vocabulario_size, d_modelo);
    token_embedding.Random();
    std::cout<<"Token Embedding size: ["<<token_embedding.fil()<<" x "<<token_embedding.col()<<"]"<<std::endl;
    std::cout<<"Capa de Embedding creada"<<std::endl;
}
void CapaEmbedding::Forward(std::vector<int>& entrada, Matriz2D& salida){
    salida.ReSize(entrada.size(), d_modelo);
    for (int i = 0; i < entrada.size(); i++){
        if (entrada[i] >= 0 && entrada[i] < token_embedding.fil()) {
            salida[i] << token_embedding[entrada[i]];
            salida[i] += posicion_embedding[i];
        } else {
            std::cout<<"Token ID Invalido: " << std::to_string(entrada[i])<<std::endl;
        }
    }
    if(embedding_matriz != &salida){
        embedding_matriz = &salida;
    }
}
CapaEmbedding::~CapaEmbedding(){
}
#endif