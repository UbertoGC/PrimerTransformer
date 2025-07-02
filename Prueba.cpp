#include <iostream>
#include "Decoder.h"
using namespace std;
int main(){
    vector<string> PruebaVector;
    string PruebaTexto = "I'll have 20";
    PruebaTexto.push_back('%');
    PruebaTexto += " of it.";
    
    string PruebaTexto2 = "GPT2 was created by OpenAI and I am using it";
    Decoder GPT1;
    GPT1.Ejecutar(PruebaTexto2);
    return 0;
}