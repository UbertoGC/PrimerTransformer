#ifndef __DECODER_H__
#define __DECODER_H__
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include "Matriz2D.h"
#include "Operaciones.h"
class Decoder
{
private:
    Matriz2D token_embedding;
    Matriz2D posicion_embedding;
    Matriz2D embedding_matriz;
    Matriz2D atencion_matriz;
    Matriz2D* capa_WQ;
    Matriz2D* capa_WK;
    Matriz2D* capa_WV;
    int vocabulario_size;
    int max_secuencia_size;
    int d_modelo;
    int d_cabeza;
    int num_cabezas;
    std::vector<int> tokens;
    std::unordered_map<std::string, int> tokenizador;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> fusiones;
public:
    Decoder();
    void Ejecutar(std::string *);
    void Atencion();
    void Embedding();
    void Tokenizacion(std::string *);
    void BytePairEncoding(std::vector<std::string> &);
    ~Decoder();
};
Decoder::Decoder(){
    //
    num_cabezas = 12;
    d_modelo = 768;
    d_cabeza = d_modelo / num_cabezas;
    char m = 92;
    char h = char(228);
    std::string wrn = "Ġ";
    std::string limit = "";
    limit.push_back(m);
    limit += "u0120";
    //CARGA DE VOCABULARIO
    std::ifstream vocabulario("vocabulary/vocab.json");
    std::string l,r;
    int i = 0;
    while (!vocabulario.eof()){
        vocabulario>>r;
        vocabulario>>l;
        if(r[0] == '{')
            r.erase(0,1);
        if(l[l.size()-1] == '}')
            l.erase(l.size()-1, 1);
        r.erase(0, 1);
        r.erase(r.size()-2, 2);
        l.erase(l.size()-1, 1);
        if(r.find(limit) != std::string::npos){
            r.erase(0,6);
            r.insert(0, 1, h);
        }
        tokenizador[r] = std::stoi(l);
    }
    vocabulario.close();
    i = 1;
    std::ifstream combinaciones("vocabulary/merges.txt");
    if(combinaciones.is_open())
    while (!combinaciones.eof()){
        combinaciones>>r;
        combinaciones>>l;
        auto ax1 = r.find(wrn);
        if(ax1 != std::string::npos){
            r.erase(ax1,wrn.size());
            r.insert(ax1, 1, h);
        }
        auto ax2 = l.find(wrn);
        if(ax2 != std::string::npos){
            l.erase(ax2,wrn.size());
            l.insert(ax2, 1, h);
        }
        fusiones[r][l] = i;
        i++;
    }
    combinaciones.close();
    vocabulario_size = tokenizador.size();
    max_secuencia_size = 512;
    std::cout<<"HOLA0"<<std::endl;
    posicion_embedding.ReSize(max_secuencia_size, d_modelo);
    std::cout<<"HOLA1"<<std::endl;
    posicion_embedding.Random();
    std::cout<<"HOLA2"<<std::endl;
    capa_WQ = new Matriz2D[num_cabezas];
    capa_WK = new Matriz2D[num_cabezas];
    capa_WV = new Matriz2D[num_cabezas];
    for (int i = 0; i < num_cabezas; i++){
        capa_WQ[i].ReSize(d_modelo, d_cabeza);
        capa_WK[i].ReSize(d_modelo, d_cabeza);
        capa_WV[i].ReSize(d_modelo, d_cabeza);
        capa_WQ[i].Random();
        capa_WK[i].Random();
        capa_WV[i].Random();
    }
    std::cout<<"HOLA3"<<std::endl;
    token_embedding.ReSize(vocabulario_size, d_modelo);
    std::cout<<"HOLA4"<<std::endl;
    token_embedding.Random();
}
void Decoder::Ejecutar(std::string *texto){
    this->Tokenizacion(texto);
    this->Embedding();
    this->Atencion();
}
void Decoder::Atencion(){
    atencion_matriz.ReSize(tokens.size(), d_modelo);
    std::cout<<"Atencion size: ["<<atencion_matriz.fil()<<" x "<<atencion_matriz.col()<<"]"<<std::endl;
    for (int c = 0; c < num_cabezas; c++){
        Matriz2D Q = embedding_matriz * capa_WQ[c];
        std::cout<<"Capa Q size: ["<<capa_WQ[c].fil()<<" x "<<capa_WQ[c].col()<<"]"<<std::endl;
        Matriz2D K = embedding_matriz * capa_WK[c];
        std::cout<<"Capa K size: ["<<capa_WK[c].fil()<<" x "<<capa_WK[c].col()<<"]"<<std::endl;
        Matriz2D V = embedding_matriz * capa_WV[c];
        std::cout<<"Capa V size: ["<<capa_WV[c].fil()<<" x "<<capa_WV[c].col()<<"]"<<std::endl;
        Matriz2D puntaje = Q * K.Transpuesta();
        puntaje *= (1.0 / sqrt(d_cabeza));
        Matriz2D mascara(tokens.size(), tokens.size(), 0);
        puntaje += mascara;
        puntaje.SoftmaxFilas();
        atencion_matriz.CopiarMatrizDatos(c, puntaje);
    }
    std::cout<<"Atencion size: ["<<atencion_matriz.fil()<<" x "<<atencion_matriz.col()<<"]"<<std::endl;
}
void Decoder::Embedding(){
    embedding_matriz.ReSize(tokens.size(), d_modelo);
    for (int i = 0; i < tokens.size(); i++){
        if (tokens[i] >= 0 && tokens[i] < token_embedding.fil()) {
            embedding_matriz[i] << token_embedding[tokens[i]];
            embedding_matriz[i] += posicion_embedding[i];
        } else {
            std::cout<<"Token ID Invalido: " << std::to_string(tokens[i])<<std::endl;
        }
    }
    std::cout<<"Embedding size: ["<<embedding_matriz.fil()<<" x "<<embedding_matriz.col()<<"]"<<std::endl;
}
void Decoder::Tokenizacion(std::string *texto){
    int s = texto->size() - 1;
    std::vector<std::string> palabras;
    std::cout<<"HOLA"<<std::endl;
    Separacion(texto, s, &palabras);
    std::cout<<"-------------------------------------------------------"<<std::endl;
    for (auto i: palabras){
        std::cout<<tokenizador[i]<<" -- "<<i<<std::endl;
    }
    std::cout<<"-------------------------------------------------------"<<std::endl;
    BytePairEncoding(palabras);
    tokens.resize(palabras.size());
    #pragma omp parallel for
    for (int i = 0; i < palabras.size();  i++){
        tokens.push_back(tokenizador[palabras[i]]);
    }
    for (int i = 0; i < palabras.size();  i++){
        std::cout<<tokens[i]<<" -- "<<i<<std::endl;
    }
    std::cout<<"Tokens size: ["<<tokens.size()<<"]"<<std::endl;
    std::cout<<"-------------------------------------------------------"<<std::endl;
}
void Decoder::BytePairEncoding(std::vector<std::string> &palabras){
    int max = palabras.size();
    for (int p = 0; p < max; p++){
        std::vector<std::string> division;
        DividirString(palabras[0],division);
        int menor = 0;
        int id = -1;
        while(division.size() > 1){
            id = -1;
            menor = 0;
            for (int i = 1; i < division.size(); ++i){
                int v = fusiones[division[i-1]][division[i]];
                if((v < menor || id == -1) && v > 0){
                    id = i;
                    menor = v;
                }
            }
            if(id > 0 && id < division.size()){
                division[id-1] += division[id];
                division.erase(division.begin() + id);
            }else{
                break;
            }
            
        }
        std::cout<<division.size()<<std::endl;
        std::cout<<palabras.size()<<std::endl;
        palabras.erase(palabras.begin());
        palabras.insert(palabras.end(),division.begin(), division.end());
    }
}
Decoder::~Decoder(){
}


#endif