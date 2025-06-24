#include <iostream>
#include "Decoder.h"
using namespace std;
int main(){
    vector<string> PruebaVector;
    string PruebaTexto = "I'll have 20";
    PruebaTexto.push_back('%');
    PruebaTexto += " of it.";
    Decoder prueba_d;
    string PruebaTexto2 = "GPT2 was created by OpenAI";
    prueba_d.Tokenizacion(&PruebaTexto2);
    /*
    ifstream a("vocabulary/vocab.json");
    map<string, int> p;
    string l,r;
    int i = 0;
    while (!a.eof()){
        a>>l;
        if(l[0] == '{')
            l.erase(0,1);
        if(l[l.size()-1] == ':' || l[l.size()-1] == ',')
            l.erase(l.size()-1, 1);
        if(l[l.size()-1] == '"')
            l.erase(l.size()-1, 1);
        if(l[0] == '"')
            l.erase(0, 1);
        if(i % 2 == 1)
            p[r] = stoi(l);
        r = l;
        i++;
    }
    for (auto i: p) {
        cout<<i.first<<" : "<<i.second<<endl;
    }
    */
    
    return 0;
}