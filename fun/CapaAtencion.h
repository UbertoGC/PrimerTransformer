#ifndef __CAPAATENCION_H__
#define __CAPAATENCION_H__

#include <cmath>
#include <iostream>
#include "Matriz2D.h"

class CapaAtencion {
private:
    Matriz2D atencion_proyeccion;
    Matriz2D atencion_datosconcat;
    Matriz2D* capa_WQ;
    Matriz2D* capa_WK;
    Matriz2D* capa_WV;
    int num_cabezas;
    int d_modelo;
    int d_cabeza;
    int t_mascara;

public:
    CapaAtencion(int n_c, int d_m, int d_c, int t_m = 0);
    void Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA = false);
    ~CapaAtencion();
};

// =======================
// Constructor
// =======================
CapaAtencion::CapaAtencion(int n_c, int d_m, int d_c, int t_m) {
    num_cabezas = n_c;
    d_modelo = d_m;
    d_cabeza = d_c;
    t_mascara = t_m;

    capa_WQ = new Matriz2D[num_cabezas];
    capa_WK = new Matriz2D[num_cabezas];
    capa_WV = new Matriz2D[num_cabezas];

    for (int i = 0; i < num_cabezas; i++) {
        capa_WQ[i].ReSize(d_modelo, d_cabeza);
        capa_WK[i].ReSize(d_modelo, d_cabeza);
        capa_WV[i].ReSize(d_modelo, d_cabeza);
        capa_WQ[i].Random();
        capa_WK[i].Random();
        capa_WV[i].Random();
    }

    atencion_proyeccion.ReSize(d_modelo, d_modelo);
    atencion_proyeccion.Random();

    std::cout << "Capa de Atencion creada con " << num_cabezas << " cabezas." << std::endl;
}

// =======================
// Forward (CPU/GPU)
// =======================
void CapaAtencion::Forward(Matriz2D& entrada, Matriz2D& salida, bool usarCUDA) {
    atencion_datosconcat.ReSize(entrada.fil(), d_modelo);
    Matriz2D mascara(entrada.fil(), entrada.fil(), t_mascara);

    int bloqueInicio = 0;
    for (int c = 0; c < num_cabezas; c++) {
        // Q, K, V
        Matriz2D Q = usarCUDA ? entrada.MultiplicarCUDA(capa_WQ[c])
                               : entrada.MultiplicarCPU(capa_WQ[c]);
        Matriz2D K = usarCUDA ? entrada.MultiplicarCUDA(capa_WK[c])
                               : entrada.MultiplicarCPU(capa_WK[c]);
        Matriz2D V = usarCUDA ? entrada.MultiplicarCUDA(capa_WV[c])
                               : entrada.MultiplicarCPU(capa_WV[c]);

        // puntaje = Q * K^T
        Matriz2D puntaje = usarCUDA ? Q.MultiplicarCUDA(K.Transpuesta())
                                     : Q.MultiplicarCPU(K.Transpuesta());

        // Escalar por 1/sqrt(d_cabeza)
        float factor = 1.0f / sqrtf((float)d_cabeza);
        if (usarCUDA){
            puntaje.EscalarCUDA(factor); // Necesitas implementar EscalarCUDA
        }
        else{
            //puntaje *= factor;
            for (int i = 0; i < puntaje.fil(); i++) {
                for (int j = 0; j < puntaje.col(); j++) {
                    puntaje(i, j) = puntaje(i, j) * factor;
                }
        }
        }


        // Sumar mascara
        if (usarCUDA){
            puntaje.SumarMatrizCUDA(mascara);
        }
        else{
            //puntaje += mascara;
            for (int i = 0; i < puntaje.fil(); i++) {
                for (int j = 0; j < puntaje.col(); j++) {
                    puntaje(i, j) = puntaje(i, j) + mascara(i, j);
                }
            }
        }
            

        // Softmax por filas
        if (usarCUDA)
            puntaje.SoftmaxFilas_CUDA();
        else
            puntaje.SoftmaxFilas();

        // output_cabeza = puntaje * V
        Matriz2D output_cabeza = usarCUDA ? puntaje.MultiplicarCUDA(V)
                                           : puntaje.MultiplicarCPU(V);

        // Concatenar en atencion_datosconcat
        atencion_datosconcat.CopiarMatrizDatos(0, bloqueInicio, output_cabeza);
        bloqueInicio += d_cabeza;
    }

    // Proyección final
    salida = usarCUDA ? atencion_datosconcat.MultiplicarCUDA(atencion_proyeccion)
                      : atencion_datosconcat.MultiplicarCPU(atencion_proyeccion);
}

// =======================
// Destructor
// =======================
CapaAtencion::~CapaAtencion() {
    delete[] capa_WQ;
    delete[] capa_WK;
    delete[] capa_WV;
}

#endif
