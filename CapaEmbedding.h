#ifndef __CAPAEMBEDDING_H__
#define __CAPAEMBEDDING_H__

#include <vector>
#include <string>
#include "Matriz2D.h"

class CapaEmbedding
{
private:
    Matriz2D posicion_embedding;
    Matriz2D* embedding_matriz;
    int d_modelo;
public:
    CapaEmbedding(int, int);
    void Forward(Matriz2D&, Matriz2D&);
    ~CapaEmbedding();
};
CapaEmbedding::CapaEmbedding(int max_entrada, int d_m){
    embedding_matriz = nullptr;
    d_modelo = d_m;
    posicion_embedding.ReSize(max_entrada, d_modelo);
    posicion_embedding.Random();
    std::cout<<"Capa de Embedding creada"<<std::endl;
}
void CapaEmbedding::Forward(Matriz2D& entrada, Matriz2D& salida){
    salida.ReSize(entrada.fil(), d_modelo);
    for (int i = 0; i < entrada.fil(); i++){
        salida[i] += posicion_embedding[i];
    }
    if(embedding_matriz != &salida){
        embedding_matriz = &salida;
    }
}
CapaEmbedding::~CapaEmbedding(){
}
#endif


/*
#ifndef __CAPAEMBEDDING_H__
#define __CAPAEMBEDDING_H__

#include <vector>
#include <string>
#include "Matriz2D.h"

class CapaEmbedding
{
private:
    Matriz2D posicion_embedding;
    Matriz2D* embedding_matriz;
    int d_modelo;
public:
    CapaEmbedding(int, int);
    void Forward(Matriz2D&, Matriz2D&);
    ~CapaEmbedding();
};
CapaEmbedding::CapaEmbedding(int max_entrada, int d_m){
    embedding_matriz = nullptr;
    d_modelo = d_m;
    posicion_embedding.ReSize(max_entrada, d_modelo);
    posicion_embedding.Random();
    std::cout<<"Capa de Embedding creada"<<std::endl;
}
void CapaEmbedding::Forward(Matriz2D& entrada, Matriz2D& salida){
    salida.ReSize(entrada.fil(), d_modelo);
    for (int i = 0; i < entrada.fil(); i++){
        salida[i] += posicion_embedding[i];
    }
    if(embedding_matriz != &salida){
        embedding_matriz = &salida;
    }
}
CapaEmbedding::~CapaEmbedding(){
}
#endif
*/