#ifndef __DECODER_H__
#define __DECODER_H__
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include "Matriz2D.h"
#include "CapaTokenizacion.h"
#include "CapaEmbedding.h"
#include "CapaAtencion.h"
class Decoder
{
private:
    CapaAtencion* atencion;
    CapaTokenizacion* tokenizador;
    CapaEmbedding* embedding;
    std::string entrada;
    std::vector<int> salida_tokenizador;
    Matriz2D salida_embedding;
    Matriz2D salida_atencion;
    Matriz2D salida_preMLP;
    int vocabulario_size;
    int max_secuencia_size;
    int d_modelo;
    int d_cabeza;
    int num_cabezas;
public:
    Decoder();
    void Ejecutar(std::string &);
    void PreMLP();
    void Atencion();
    void Embedding();
    void Tokenizacion();
    ~Decoder();
};
Decoder::Decoder(){
    num_cabezas = 12;
    d_modelo = 768;
    d_cabeza = d_modelo / num_cabezas;
    max_secuencia_size = 512;
    tokenizador = new CapaTokenizacion();
    vocabulario_size = tokenizador->VocabSize();
    embedding = new CapaEmbedding(vocabulario_size, d_modelo);
    atencion = new CapaAtencion(num_cabezas, d_modelo, d_cabeza);
}
void Decoder::Ejecutar(std::string &texto){
    if(texto.size()){
        std::cerr<<"Error: Entrada vacia."<<std::endl;
        return;
    }
    if(texto.size() > max_secuencia_size){
        std::cerr<<"Error: Entrada excede el maximo de caracteres permitidos."<<std::endl;
        return;
    }
    entrada = texto;
    this->Tokenizacion();
    this->Embedding();
    this->Atencion();
    this->PreMLP();
}
void Decoder::PreMLP(){
    salida_preMLP = salida_atencion + salida_embedding;
    salida_preMLP.NormalizarFilas();
    std::cout<<"PreMLP size: ["<<salida_preMLP.fil()<<" x "<<salida_preMLP.col()<<"]"<<std::endl;
}
void Decoder::Atencion(){
    atencion->Forward(salida_embedding, salida_atencion);
    std::cout<<"Atencion size: ["<<salida_atencion.fil()<<" x "<<salida_atencion.col()<<"]"<<std::endl;
}
void Decoder::Embedding(){
    embedding->Forward(salida_tokenizador, salida_embedding);
    std::cout<<"Embedding size: ["<<salida_embedding.fil()<<" x "<<salida_embedding.col()<<"]"<<std::endl;
}
void Decoder::Tokenizacion(){
    tokenizador->Forward(entrada, salida_tokenizador);
    std::cout<<"Tokens size: "<<salida_tokenizador.size()<<std::endl;
}
Decoder::~Decoder(){
    delete tokenizador;
    delete embedding;
    delete atencion;
}


#endif