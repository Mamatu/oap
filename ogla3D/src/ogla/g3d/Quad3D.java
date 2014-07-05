/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.awt.Color;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class Quad3D {

    public Vertex3D[] vertices = {new Vertex3D(), new Vertex3D(), new Vertex3D(), new Vertex3D()};

    public void setColor(Color color) {
        for (Vertex3D v : vertices) {
            v.color = color;
        }
    }

    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("[");
        for (Vertex3D v : this.vertices) {
            stringBuilder.append("[");
            stringBuilder.append(String.valueOf(v.position.x));
            stringBuilder.append(",");
            stringBuilder.append(String.valueOf(v.position.y));
            stringBuilder.append(",");
            stringBuilder.append(String.valueOf(v.position.z));
            stringBuilder.append("]");
        }
        stringBuilder.append("]");
        return stringBuilder.toString();
    }
}
