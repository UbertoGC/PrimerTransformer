#ifndef __DECODER_H__
#define __DECODER_H__
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include "Matriz2D.h"
#include "CapaTokenizacion.h"
#include "CapaEmbedding.h"
#include "BloqueTransformer.h"
class Decoder
{
private:
    CapaTokenizacion* tokenizador;
    CapaEmbedding* embedding;
    BloqueTransformer** bloques;
    std::string entrada;
    Matriz2D salida_tokenizador;
    Matriz2D salida_embedding;
    Matriz2D* salida_bloques;
    int num_bloques;
    int max_secuencia_size;
    int d_modelo;
    int d_cabeza;
    int d_feedforward;
    int num_cabezas;
public:
    Decoder();
    void Ejecutar(std::string&);
    ~Decoder();
};
Decoder::Decoder(){
    std::cout<<"Creando Decoder..."<<std::endl;
    num_cabezas = 12;
    d_modelo = 768;
    d_cabeza = d_modelo / num_cabezas;
    d_feedforward = 3072;
    num_bloques = 12;
    max_secuencia_size = 512;
    tokenizador = new CapaTokenizacion(d_modelo);
    embedding = new CapaEmbedding(max_secuencia_size, d_modelo);
    bloques = new BloqueTransformer*[num_bloques];
    for (int i = 0; i < num_bloques; i++){
        std::cout<<i+1<<". ";
        bloques[i] = new BloqueTransformer();
    }
    salida_bloques = new Matriz2D[num_bloques];
}
void Decoder::Ejecutar(std::string& texto){
    entrada = texto;
    this->tokenizador->Forward(entrada, salida_tokenizador);
    std::cout<<"Tonekizador size: ["<<salida_tokenizador.fil()<<" x "<<salida_tokenizador.col()<<"]"<<std::endl;
    this->embedding->Forward(salida_tokenizador, salida_embedding);
    std::cout<<"Embedding size: ["<<salida_embedding.fil()<<" x "<<salida_embedding.col()<<"]"<<std::endl;
    this->bloques[0]->Forward(salida_embedding,salida_bloques[0]);
    std::cout<<"Bloque 1 size: ["<<salida_bloques[0].fil()<<" x "<<salida_bloques[0].col()<<"]"<<std::endl;
    for (int i = 1; i < num_bloques; i++){
        this->bloques[i]->Forward(salida_bloques[i-1],salida_bloques[i]);
        std::cout<<"Bloque "<<i+1<<" size: ["<<salida_bloques[i].fil()<<" x "<<salida_bloques[i].col()<<"]"<<std::endl;
    }
}
Decoder::~Decoder(){
    delete embedding;
    for (int i = 0; i < num_bloques; i++){
        delete bloques[i];
    }
    delete[] bloques;
}
#endif