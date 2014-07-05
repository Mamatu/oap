/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d.utils;

import java.awt.Color;
import ogla.core.util.Writer;
import ogla.g3d.Line3D;
import ogla.g3d.Object3D;
import ogla.g3d.Quad3D;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class Tools3D {

    public static void transformColor(Vector3 vec, Color color) {
        vec.set(color.getRed(), color.getGreen(), color.getBlue());
    }

    public static void quadsToBytes(Object3D object3D, Writer vertexs, Writer colors) {
        for (int fa = 0; fa < object3D.getQuads3DCount(); fa++) {
            Quad3D quad = object3D.getQuad3D(fa);
            for (int fb = 0; fb < 4; fb++) {
                vertexs.write(quad.vertices[fb].position.x);
                vertexs.write(quad.vertices[fb].position.y);
                vertexs.write(quad.vertices[fb].position.z);
                Integer i = quad.vertices[fb].color.getRed();
                colors.write(i.floatValue() / 255.f);
                i = quad.vertices[fb].color.getGreen();
                colors.write(i.floatValue() / 255.f);
                i = quad.vertices[fb].color.getBlue();
                colors.write(i.floatValue() / 255.f);
                i = quad.vertices[fb].color.getAlpha();
                colors.write(i.byteValue() / 255.f);
            }
        }
    }

    public static void linesToBytes(Object3D object3D, Writer vertexs, Writer colors) {
        for (int fa = 0; fa < object3D.getLines3DCount(); fa++) {
            Line3D line = object3D.getLine3D(fa);
            for (int fb = 0; fb < 2; fb++) {
                vertexs.write(line.vertices[fb].position.x);
                vertexs.write(line.vertices[fb].position.y);
                vertexs.write(line.vertices[fb].position.z);
                Integer i = line.vertices[fb].color.getRed();
                colors.write(i.floatValue() / 255.f);
                i = line.vertices[fb].color.getGreen();
                colors.write(i.floatValue() / 255.f);
                i = line.vertices[fb].color.getBlue();
                colors.write(i.floatValue() / 255.f);
                i = line.vertices[fb].color.getAlpha();
                colors.write(i.byteValue() / 255.f);
            }
        }
    }
}
