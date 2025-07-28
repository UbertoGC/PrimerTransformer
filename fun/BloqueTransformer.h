#ifndef __BLOQUE_TRANSFORMER_H__
#define __BLOQUE_TRANSFORMER_H__

#include "CapaAtencion.h"
#include "CapaFeedForward.h"
#include "Matriz2D.h"

class BloqueTransformer {
private:
    CapaAtencion* atencion;
    CapaFeedForward* feedforward;

    Matriz2D gamma1, beta1; // Para Normalización antes de atención
    Matriz2D gamma2, beta2; // Para Normalización antes de feedforward

public:
    BloqueTransformer(int d_ff, int n_c, int d_m, int d_c, int t_m);
    BloqueTransformer();
    void Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA = false);
    ~BloqueTransformer();
};

// Constructor
BloqueTransformer::BloqueTransformer(int d_ff, int n_c, int d_m, int d_c, int t_m) {
    atencion = new CapaAtencion(n_c, d_m, d_c, t_m);
    feedforward = new CapaFeedForward(d_ff, d_m);

    // Inicializar parámetros de normalización
    gamma1.ReSize(1, d_m);
    beta1.ReSize(1, d_m);
    gamma2.ReSize(1, d_m);
    beta2.ReSize(1, d_m);

    gamma1.Fill(1.0f); // γ = 1
    beta1.Fill(0.0f);  // β = 0
    gamma2.Fill(1.0f);
    beta2.Fill(0.0f);

    std::cout << "Bloque Transformer creado (d_m=" << d_m << ", d_ff=" << d_ff << ")" << std::endl;
}
BloqueTransformer::BloqueTransformer(){
    gamma1.ReSize(1, 768);
    beta1.ReSize(1, 768);
    gamma2.ReSize(1, 768);
    beta2.ReSize(1, 768);
    gamma1.Fill(1.0f);
    beta1.Fill(0.0f);
    gamma2.Fill(1.0f);
    beta2.Fill(0.0f);

    atencion = new CapaAtencion(12, 768, 64, 0);
    feedforward = new CapaFeedForward(3072, 768);
    std::cout<<"Bloque Transformer creada"<<std::endl;
}
// Forward con GPU opcional
void BloqueTransformer::Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA) {
    // === 1. Normalización antes de Atención ===
    Matriz2D norm1 = entrada;
    if (usarCUDA)
        norm1.NormalizarFilas_CUDA(gamma1, beta1); // Implementar esta versión en GPU
    else
        norm1.NormalizarFilas(gamma1, beta1);

    // === 2. Atención Multi-Cabeza ===
    Matriz2D salida_atencion;
    atencion->Forward(norm1, salida_atencion, usarCUDA);

    // Residual
    //salida_atencion += entrada;
    for (int i = 0; i < salida_atencion.Filas(); i++) {
        for (int j = 0; j < salida_atencion.Columnas(); j++) {
            salida_atencion(i, j) += entrada(i, j);
        }
    }

    // === 3. Normalización antes de FeedForward ===
    Matriz2D norm2 = salida_atencion;
    if (usarCUDA)
        norm2.NormalizarFilas_CUDA(gamma2, beta2);
    else
        norm2.NormalizarFilas(gamma2, beta2);

    // === 4. FeedForward ===
    feedforward->Forward(norm2, salida, usarCUDA);

    // Residual
    //salida += salida_atencion;
    for (int i = 0; i < salida.fil(); i++) {
        for (int j = 0; j < salida.col(); j++) {
            salida(i, j) += salida_atencion(i, j);
        }
    }
}

// Destructor
BloqueTransformer::~BloqueTransformer() {
    delete atencion;
    delete feedforward;
}

#endif
