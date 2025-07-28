#ifndef __CAPAFEEDFORWARD_H__
#define __CAPAFEEDFORWARD_H__

#include <iostream>
#include "Matriz2D.h"

class CapaFeedForward {
private:
    Matriz2D pesos1;  // [d_m x d_ff]
    Matriz2D pesos2;  // [d_ff x d_m]
    Matriz2D bias1;   // [1 x d_ff]
    Matriz2D bias2;   // [1 x d_m]

public:
    CapaFeedForward(int d_ff, int d_m);
    void Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA = false);
    ~CapaFeedForward();
};

CapaFeedForward::CapaFeedForward(int d_ff, int d_m) {
    pesos1.ReSize(d_m, d_ff);
    pesos2.ReSize(d_ff, d_m);
    bias1.ReSize(1, d_ff);
    bias2.ReSize(1, d_m);

    pesos1.Random();
    pesos2.Random();
    bias1.Random();
    bias2.Random();

    std::cout << "Capa FeedForward creada con dimensiones: "
              << "[d_m=" << d_m << ", d_ff=" << d_ff << "]" << std::endl;
}

void CapaFeedForward::Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA) {
    if (entrada.col() != pesos1.fil()) {
        throw std::runtime_error("Dimensiones incompatibles en Forward()");
    }

    // Primera proyección: entrada * W1
    // Primera proyección
    Matriz2D salida_ff = usarCUDA ? entrada.MultiplicarCUDA(pesos1)
                                : entrada.MultiplicarCPU(pesos1);


    // Sumar bias1 (broadcast)
    if (usarCUDA)
        salida_ff.SumarFilaCUDA(bias1);
    else
        salida_ff.SumarFila(bias1);


    // Activación
    if (usarCUDA)
        salida_ff.RELU_CUDA();
    else
        salida_ff.RELU();

    // Segunda proyección: salida_ff * W2
    // Segunda proyección
    salida = usarCUDA ? salida_ff.MultiplicarCUDA(pesos2)
                    : salida_ff.MultiplicarCPU(pesos2);

    // Sumar bias2 (broadcast)
    if (usarCUDA)
        salida.SumarFilaCUDA(bias2);
    else
        salida.SumarFila(bias2);
   
    // Activación final (nova de acuerdo alestandard Transformer)
    if (usarCUDA)
        salida.RELU_CUDA();
    else
        salida.RELU();
}

CapaFeedForward::~CapaFeedForward() {}

#endif
