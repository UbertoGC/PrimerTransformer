#ifndef __BLOQUE_TRANSFORMER_H__
#define __BLOQUE_TRANSFORMER_H__
#include "CapaAtencion.h"
#include "CapaFeedForward.h"
class BloqueTransformer {
private:
    CapaAtencion* atencion;
    CapaFeedForward* feedforward;
public:
    BloqueTransformer();
    BloqueTransformer(int,int, int, int, int);
    void Forward(Matriz2D& entrada, Matriz2D& salida);
    ~BloqueTransformer();
};
BloqueTransformer::BloqueTransformer(){
    atencion = new CapaAtencion(12, 768, 64, 0);
    feedforward = new CapaFeedForward(3072, 768);
    std::cout<<"Bloque Transformer creada"<<std::endl;
}
BloqueTransformer::BloqueTransformer(int d_ff, int n_c, int d_m, int d_c, int t_m) {
    atencion = new CapaAtencion(n_c, d_m, d_c, t_m);
    feedforward = new CapaFeedForward(d_ff, d_m);
    std::cout<<"Bloque Transformer creada"<<std::endl;
}
void BloqueTransformer::Forward(Matriz2D& entrada, Matriz2D& salida) {
    Matriz2D salida_atencion;
    
    atencion->Forward(entrada, salida_atencion);
    salida_atencion += entrada;
    salida_atencion.NormalizarFilas();

    feedforward->Forward(salida_atencion, salida);
    salida += salida_atencion;
    salida.NormalizarFilas();
}
BloqueTransformer::~BloqueTransformer() {
    delete atencion;
    delete feedforward;
}
#endif