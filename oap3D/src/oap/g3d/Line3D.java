/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.awt.Color;
import ogla.g3d.utils.Tools3D;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class Line3D {

    public Vertex3D[] vertices = {new Vertex3D(), new Vertex3D()};
    public float thickness = 1.f;

    public Line3D() {
    }

    public Line3D(float x, float y, float z, float x1, float y1, float z1) {
        this.set(x, y, z, x1, y1, z1);
    }

    public Line3D(Color color) {
        this.setColor(color);
    }

    private Vector3 temp = new Vector3();

    public void setColor(Color color) {
        this.vertices[0].color = color;
        this.vertices[1].color = color;
    }

    public void set(Line3D line3D) {
        this.vertices[0].position.set(line3D.vertices[0].position);
        this.vertices[1].position.set(line3D.vertices[1].position);
        this.vertices[0].color = line3D.vertices[0].color;
        this.vertices[1].color = line3D.vertices[0].color;
        this.thickness = line3D.thickness;
    }

    public void setVector(Line3D line3D) {
        this.vertices[0].position.set(line3D.vertices[0].position);
        this.vertices[1].position.set(line3D.vertices[1].position);
    }

    public void translate(Vector3 v) {
        this.vertices[0].position.add(v);
        this.vertices[1].position.add(v);
    }

    void set(float x, float y, float z, float x1, float y1, float z1) {
        this.vertices[0].position.x = x;
        this.vertices[0].position.y = y;
        this.vertices[0].position.z = z;
        this.vertices[1].position.x = x1;
        this.vertices[1].position.y = y1;
        this.vertices[1].position.z = z1;
    }
}
