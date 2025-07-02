#ifndef __CAPAFEEDFORWARD_H__
#define __CAPAFEEDFORWARD_H__
#include <vector>
#include <string>
#include "Matriz2D.h"
class CapaFeedForward {
private:
    Matriz2D pesos1;
    Matriz2D pesos2;
    Vector2D bias1;
    Vector2D bias2;
public:
    CapaFeedForward(int, int);
    void Forward(Matriz2D& entrada, Matriz2D& salida);
    ~CapaFeedForward();
};
CapaFeedForward::CapaFeedForward(int d_ff, int d_m) {
    pesos1.ReSize(d_m, d_ff);
    pesos2.ReSize(d_ff, d_m);
    bias1.ReSize(d_ff);
    bias2.ReSize(d_m);
    pesos1.Random();
    pesos2.Random();
    bias1.Random();
    bias2.Random();
    std::cout << "Capa FeedForward creada" << std::endl;
}
void CapaFeedForward::Forward(Matriz2D& entrada, Matriz2D& salida) {
    Matriz2D salida_ff = entrada * pesos1;
    salida_ff += bias1;
    salida_ff.RELU();
    //std::cout << "Salida FF size: [" << salida_ff.fil() << " x " << salida_ff.col() << "]" << std::endl;
    salida = salida_ff * pesos2;
    salida += bias2;
    salida.RELU();
}
CapaFeedForward::~CapaFeedForward() {
}
#endif