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
    int d_modelo;
    Matriz2D embedding_matriz;
    std::vector<int> tokens;
    std::unordered_map<std::string, int> tokenizador;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> fusiones;
public:
    Decoder();
    void Embeding();
    void Tokenizacion(std::string *);
    void BytePairEncoding(std::vector<std::string> &);
    ~Decoder();
};

Decoder::Decoder(){
    //
    d_modelo = 768;
    //OMITIR \u0120
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
}
void Decoder::Embeding(){
    embedding_matriz.ReSize(tokens.size(), d_modelo);
    for (int i = 0; i < tokens.size(); i++){
        if (tokens[i] >= 0 && tokens[i] < token_embedding.fil()) {
            embedding_matriz[i] << token_embedding[i];
            embedding_matriz[i] += posicion_embedding[i];
        } else {
            std::cout<<"Token ID Invalido: " << std::to_string(tokens[i])<<std::endl;
        }
    }
    embedding_matriz.Transponer();
}
void Decoder::Tokenizacion(std::string *texto){
    int s = texto->size() - 1;
    std::vector<std::string> palabras;
    Separacion(texto, s, &palabras);
    std::cout<<"-------------------------------------------------------"<<std::endl;
    for (auto i: palabras){
        std::cout<<tokenizador[i]<<" -- "<<i<<std::endl;
    }
    std::cout<<"-------------------------------------------------------"<<std::endl;
    BytePairEncoding(palabras);
    for (int i = 0; i < palabras.size();  i++){
        tokens[i] = tokenizador[palabras[i]];
        std::cout<<tokens[i]<<" -- "<<palabras[i]<<std::endl;
    }
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
            for (int i = 1; i < division.size(); i++){
                int v = fusiones[division[i-1]][division[i]];
                if((v < menor || id == -1) && v > 0){
                    id = i;
                    menor = v;
                }
            }
            if(id != -1){
                division[id-1] += division[id];
                division.erase(division.begin() + id);
            }else{
                break;
            }
        }
        palabras.insert(palabras.end(),division.begin(), division.end());
        palabras.erase(palabras.begin());
    }
}
Decoder::~Decoder(){
}


#endif