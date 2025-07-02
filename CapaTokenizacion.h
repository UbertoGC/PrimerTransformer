#ifndef __CAPATOKENIZACION_H__
#define __CAPATOKENIZACION_H__
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
class CapaTokenizacion{
private:
    Matriz2D token_embedding;
    Matriz2D* tokenizacion_matriz;
    std::vector<int> palabras_token;
    std::unordered_map<std::string, int> tokenizador;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> fusiones;
public:
    CapaTokenizacion(int);
    void Forward(std::string &, Matriz2D&);
    void BytePairEncoding(std::vector<std::string> &);
    void Separacion(std::string&, int, std::vector<std::string> &);
    void DividirString(std::string &, std::vector<std::string> &);
    int VocabSize();
    ~CapaTokenizacion();
};
CapaTokenizacion::CapaTokenizacion(int d_modelo){
    char m = 92;
    char h = char(228);
    std::string wrn = "Ġ";
    std::string limit = "";
    limit.push_back(m);
    limit += "u0120";
    tokenizacion_matriz = nullptr;
    
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
    token_embedding.ReSize(tokenizador.size(), d_modelo);
    token_embedding.Random();
    std::cout<<"Token Embedding size: ["<<token_embedding.fil()<<" x "<<token_embedding.col()<<"]"<<std::endl;
    std::cout<<"Capa de Tokenizacion creada"<<std::endl;
}
void CapaTokenizacion::Forward(std::string &entrada, Matriz2D& salida){
    int s = entrada.size() - 1;
    std::vector<std::string> palabras;
    Separacion(entrada, s, palabras);
    BytePairEncoding(palabras);
    palabras_token.clear();
    #pragma omp parallel for
    for (int i = 0; i < palabras.size();  i++){
        palabras_token.push_back(tokenizador[palabras[i]]);
    }
    salida.ReSize(palabras.size(), token_embedding.col());
    for (int i = 0; i < palabras_token.size(); i++){
        if (palabras_token[i] >= 0 && palabras_token[i] < token_embedding.fil()) {
            salida[i] << token_embedding[palabras_token[i]];
        } else {
            std::cout<<"Token ID Invalido: " << palabras_token[i]<<std::endl;
        }
    }
    if(tokenizacion_matriz != &salida){
        tokenizacion_matriz = &salida;
    }
}
void CapaTokenizacion::BytePairEncoding(std::vector<std::string> &palabras){
    int max = palabras.size();
    for (int p = 0; p < max; p++){
        std::vector<std::string> division;
        DividirString(palabras[0],division);
        int menor = 0;
        int id = -1;
        while(division.size() > 1){
            id = -1;
            menor = INT_MAX;
            for (int i = 1; i < division.size(); ++i){
                int v = fusiones[division[i-1]][division[i]];
                if((v < menor) && v > 0){
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
        palabras.erase(palabras.begin());
        palabras.insert(palabras.end(),division.begin(), division.end());
    }
}
void CapaTokenizacion::Separacion(std::string& texto, int s, std::vector<std::string>& r){
    std::string p = "";
    std::string d;
    d.push_back(char(228));
    for (int i = 0; i <= s; i++){
        if(texto[i] == ' '){
            if(p != "" && p != d){
                r.push_back(p);
                p = d;
            }
        }
        else if(texto[i] == char(39)){
            if(p != "" && p != "'" && p != d){
                r.push_back(p);
                p = "'";
            }
        }
        else{
            if(isdigit(texto[i]) || isalpha(texto[i])){
                if(p != "" && p != d){
                    if(!(isdigit(p[p.size()-1])) && !(isalpha(p[p.size()-1]))){
                        r.push_back(p);
                        p = "";
                    }
                }
            }
            else{
                if(p != "" && p != d){
                    if((isalpha(p[p.size()-1]) || isdigit(p[p.size()-1]))){
                        r.push_back(p);
                        p = "";
                    }
                }
            }
            p.push_back(texto[i]);
        }
    }
    if(p != "" && p != d)
        r.push_back(p);
}
void CapaTokenizacion::DividirString(std::string &palabra, std::vector<std::string> &division){
    for (char letra:palabra){
        std::string l = "";
        l.push_back(letra);
        division.push_back(l);
    }
}
int CapaTokenizacion::VocabSize(){
    return tokenizador.size();
}
CapaTokenizacion::~CapaTokenizacion(){
}
#endif