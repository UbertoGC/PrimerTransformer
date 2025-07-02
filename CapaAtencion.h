#ifndef __CAPAATENCION_H__
#define __CAPAATENCION_H__

#include <vector>
#include <string>
#include "Matriz2D.h"

class CapaAtencion
{
protected:
    Matriz2D atencion_proyeccion;
    Matriz2D atencion_datosconcat;
    Matriz2D* capa_WQ;
    Matriz2D* capa_WK;
    Matriz2D* capa_WV;
    Matriz2D* atencion_matriz;
    int num_cabezas;
    int d_modelo;
    int d_cabeza;
    int t_mascara;
public:
    CapaAtencion(int, int, int, int t_m = 0);
    void Forward(Matriz2D&, Matriz2D&);
    ~CapaAtencion();
};
CapaAtencion::CapaAtencion(int n_c, int d_m, int d_c, int t_m) {
    num_cabezas = n_c;
    d_modelo = d_m;
    d_cabeza = d_c;
    t_mascara = t_m;

    capa_WQ = new Matriz2D[num_cabezas];
    capa_WK = new Matriz2D[num_cabezas];
    capa_WV = new Matriz2D[num_cabezas];
    for (int i = 0; i < num_cabezas; i++){
        capa_WQ[i].ReSize(d_modelo, d_cabeza);
        capa_WK[i].ReSize(d_modelo, d_cabeza);
        capa_WV[i].ReSize(d_modelo, d_cabeza);
        capa_WQ[i].Random();
        capa_WK[i].Random();
        capa_WV[i].Random();
    }
    atencion_matriz = nullptr;
    atencion_proyeccion.ReSize(d_modelo, d_modelo);
    atencion_proyeccion.Random();
    std::cout<<"Capa de Atencion creada"<<std::endl;
}
void CapaAtencion::Forward(Matriz2D& entrada, Matriz2D& salida) {
    atencion_datosconcat.ReSize(entrada.fil(), d_modelo);
    Matriz2D mascara(entrada.fil(), entrada.fil(), t_mascara);
    for (int c = 0; c < num_cabezas; c++) {
        Matriz2D Q = entrada * capa_WQ[c];
        Matriz2D K = entrada * capa_WK[c];
        Matriz2D V = entrada * capa_WV[c];
        Matriz2D puntaje = Q * K.Transpuesta();
        puntaje *= (1.0 / sqrt(d_cabeza));
        puntaje += mascara;
        puntaje.SoftmaxFilas();
        Matriz2D output_cabeza = puntaje * V;
        atencion_datosconcat.CopiarMatrizDatos(0, c, output_cabeza);
    }
    salida = atencion_datosconcat * atencion_proyeccion;
    if (atencion_matriz != &salida) {
        atencion_matriz = &salida;
    }
}
CapaAtencion::~CapaAtencion() {
    delete[] capa_WQ;
    delete[] capa_WK;
    delete[] capa_WV;
}
#endif