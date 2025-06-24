#ifndef __OPERACIONES_H__
#define __OPERACIONES_H__
#include <vector>
#include <string>
#include <map>
#include <math.h>
void Softmax(double* z, int s, double* r){
    double suma = 0;
    for (int i = 0; i < s; i++){
        r[i] =  pow(2.718281, z[i]);
        suma += r[i];
    }
    for (int i = 0; i < s; i++){
        r[i] = r[i]/suma;
    }
}
void Separacion(std::string* texto, int s, std::vector<std::string>* r){
    std::string p = "";
    std::string d;
    d.push_back(char(228));
    for (int i = 0; i <= s; i++){
        if((*texto)[i] == ' '){
            if(p != "" && p != d){
                r->push_back(p);
                p = d;
            }
        }
        else if((*texto)[i] == char(39)){
            if(p != "" && p != "'" && p != d){
                r->push_back(p);
                p = "'";
            }
        }
        else{
            if(isdigit((*texto)[i]) || isalpha((*texto)[i])){
                if(p != "" && p != d){
                    if(!(isdigit(p[p.size()-1])) && !(isalpha(p[p.size()-1]))){
                        r->push_back(p);
                        p = "";
                    }
                }
            }
            else{
                if(p != "" && p != d){
                    if((isalpha(p[p.size()-1]) || isdigit(p[p.size()-1]))){
                        r->push_back(p);
                        p = "";
                    }
                }
            }
            p.push_back((*texto)[i]);
        }
    }
    if(p != "" && p != d)
        r->push_back(p);
}
void DividirString(std::string &palabra, std::vector<std::string> &division){
    for (char letra:palabra){
        std::string l = "";
        l.push_back(letra);
        division.push_back(l);
    }
}
#endif