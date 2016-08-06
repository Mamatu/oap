/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.util.ArrayList;
import java.util.List;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class Object3D {

    private List<Quad3D> quads = new ArrayList<Quad3D>();
    private List<Line3D> lines = new ArrayList<Line3D>();
    private Vector3 position = new Vector3();

    public void addQuad(Quad3D quad) {
        quads.add(quad);
    }

    public void removeQuad3D(Quad3D quad) {
        quads.remove(quad);
    }

    public void addLine3D(Line3D line) {
        this.lines.add(line);
    }

    public void removeLine(Line3D line) {
        this.lines.remove(line);
    }

    public Quad3D getQuad3D(int index) {
        return quads.get(index);
    }

    public int getQuads3DCount() {
        return quads.size();
    }

    public void setPosition(Vector3 vec) {
        for (Quad3D quad : quads) {
            for (Vertex3D v : quad.vertices) {
                v.position.x = vec.x - this.position.x;
                v.position.y = vec.y - this.position.y;
                v.position.z = vec.z - this.position.z;
            }
        }
        this.position.set(vec);
    }

    public Vector3 getPosition(Vector3 vec) {
        vec.set(this.position);
        return vec;
    }

    public Line3D getLine3D(int index) {
        return lines.get(index);
    }

    public int getLines3DCount() {
        return lines.size();
    }

}
