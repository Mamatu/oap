/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import javax.media.opengl.GL;
import javax.media.opengl.glu.GLU;
import ogla.math.Quaternion;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public interface Renderer {

    public void addObject3D(Object3D object3D);

    public void removeObject3D(Object3D object3D);

    public void updateObject3DGeometry(Object3D object3D);

    public void scale(Vector3 vec, Object3D object3D);

    public void translate(Vector3 vec, Object3D object3D);

    public void rotate(Quaternion quaternion, Object3D object3D);

    public void render(GL gl, GLU glu);

}
