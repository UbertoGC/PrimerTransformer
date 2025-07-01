#include <iostream>
#include "Decoder.h"
using namespace std;
int main(){
    vector<string> PruebaVector;
    string PruebaTexto = "I'll have 20";
    PruebaTexto.push_back('%');
    PruebaTexto += " of it.";
    
    string PruebaTexto2 = "GPT2 was created by OpenAI";
    std::cout<<"HOLA"<<std::endl;
    Decoder prueba_d;
    prueba_d.Ejecutar(&PruebaTexto2);

    return 0;
}