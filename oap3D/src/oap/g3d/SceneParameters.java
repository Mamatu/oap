/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import ogla.math.Vector3;

public class SceneParameters {

    public float x0 = 0;
    public float y0 = 0;
    public float z0 = 0;
    public float width = 10;
    public float height = 10;
    public float deep = 10;
    public float stepx = 0.2f;
    public float stepy = 0.2f;

    public SceneParameters() {
    }

    public SceneParameters(float x0, float y0, float z0, float width, float height, float deep) {
        this(x0, y0, z0, width, height, deep, 0.2f, 0.2f);
    }

    public SceneParameters(float x0, float y0, float z0, float width, float height, float deep, float stepx, float stepy) {
        this.x0 = x0;
        this.y0 = y0;
        this.z0 = z0;
        this.width = width;
        this.height = height;
        this.deep = deep;
        this.stepx = stepx;
        this.stepy = stepy;
    }

    public float getX0() {
        return this.x0;
    }

    public float getX1() {
        return this.x0 + width;
    }

    public float getY0() {
        return this.y0;
    }

    public float getY1() {
        return this.y0 + height;
    }

    public float getZ0() {
        return this.z0;
    }

    public float getZ1() {
        return this.z0 + deep;
    }

    public static SceneParameters getSceneParameters(Object3D object3D, SceneParameters parameters) {
        Float minx = null;
        Float miny = null;
        Float minz = null;
        Float maxx = null;
        Float maxy = null;
        Float maxz = null;
        for (int fa = 0; fa < object3D.getLines3DCount(); fa++) {
            Line3D line3D = object3D.getLine3D(fa);
            for (Vertex3D ve : line3D.vertices) {
                Vector3 v = ve.position;
                minx = (minx == null || v.x < minx) ? v.x : minx;
                miny = (miny == null || v.y < miny) ? v.y : miny;
                minz = (minz == null || v.z < minz) ? v.z : minz;
                maxx = (maxx == null || v.x > maxx) ? v.x : maxx;
                maxy = (maxy == null || v.y > maxy) ? v.y : maxy;
                maxz = (maxz == null || v.z > maxz) ? v.z : maxz;
            }
        }
        for (int fa = 0; fa < object3D.getQuads3DCount(); fa++) {
            Quad3D quad3D = object3D.getQuad3D(fa);
            for (Vertex3D ve : quad3D.vertices) {
                Vector3 v = ve.position;
                minx = (minx == null || v.x < minx) ? v.x : minx;
                miny = (miny == null || v.y < miny) ? v.y : miny;
                minz = (minz == null || v.z < minz) ? v.z : minz;
                maxx = (maxx == null || v.x > maxx) ? v.x : maxx;
                maxy = (maxy == null || v.y > maxy) ? v.y : maxy;
                maxz = (maxz == null || v.z > maxz) ? v.z : maxz;
            }
        }
        parameters.x0 = minx;
        parameters.width = maxx - minx;
        parameters.y0 = miny;
        parameters.height = maxy - miny;
        parameters.z0 = minz;
        parameters.deep = maxz - minz;
        return parameters;
    }
}
