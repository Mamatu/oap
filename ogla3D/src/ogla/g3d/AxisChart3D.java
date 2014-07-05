/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.media.opengl.GL;
import javax.media.opengl.glu.GLU;
import javax.swing.JPanel;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class AxisChart3D extends Canvas3D {

    public AxisChart3D(Renderer renderer, Camera camera, JPanel jPanel) {
        super(renderer, camera, jPanel);
    }

    private class CoordSystem3D extends Object3D {

        public CoordSystem cs = null;

        public CoordSystem3D(CoordSystem cs) {
            this.cs = cs;
            this.addLine3D(cs.axisX);
            this.addLine3D(cs.axisY);
            this.addLine3D(cs.axisZ);
        }
    }

    private class Box3D extends Object3D {

        public Box3D() {
            for (int fa = 0; fa < 12; fa++) {
                this.addLine3D(new Line3D());
            }
        }

        public void update() {
            Float minx = Collections.min(AxisChart3D.this.minxs);
            Float miny = Collections.min(AxisChart3D.this.minys);
            Float minz = Collections.min(AxisChart3D.this.minzs);
            Float maxx = Collections.max(AxisChart3D.this.maxxs);
            Float maxy = Collections.max(AxisChart3D.this.maxys);
            Float maxz = Collections.max(AxisChart3D.this.maxzs);

            this.getLine3D(0).set(minx, miny, minz, maxx, miny, minz);
            this.getLine3D(1).set(minx, maxy, minz, maxx, maxy, minz);
            this.getLine3D(2).set(minx, maxy, maxz, maxx, maxy, maxz);
            this.getLine3D(3).set(minx, miny, maxz, maxx, miny, maxz);

            this.getLine3D(4).set(minx, miny, minz, minx, maxy, minz);
            this.getLine3D(5).set(minx, miny, maxz, minx, maxy, maxz);
            this.getLine3D(6).set(maxx, miny, maxz, maxx, maxy, maxz);
            this.getLine3D(7).set(maxx, miny, minz, maxx, maxy, minz);

            this.getLine3D(8).set(minx, miny, minz, minx, miny, maxz);
            this.getLine3D(9).set(maxx, miny, minz, maxx, miny, maxz);
            this.getLine3D(10).set(maxx, maxy, minz, maxx, maxy, maxz);
            this.getLine3D(11).set(minx, maxy, minz, minx, maxy, maxz);
            if (AxisChart3D.this.wasAdded) {
                AxisChart3D.super.updateObject3DGeometry(box3D);
            }
        }
    }

    private Box3D box3D = new Box3D();

    public void setDefaultCoordSystem(float length) {
        CoordSystem coordSystem = new CoordSystem();
        coordSystem.axisX.thickness = 5.f;
        coordSystem.axisY.thickness = 5.f;
        coordSystem.axisZ.thickness = 5.f;
        coordSystem.axisX.vertices[1].position.x = 1 * length;
        coordSystem.axisX.vertices[1].position.y = 0;
        coordSystem.axisX.vertices[1].position.z = 0;
        coordSystem.axisX.setColor(Color.red);
        coordSystem.axisY.vertices[1].position.x = 0;
        coordSystem.axisY.vertices[1].position.y = 1 * length;
        coordSystem.axisY.vertices[1].position.z = 0;
        coordSystem.axisY.setColor(Color.green);
        coordSystem.axisZ.vertices[1].position.x = 0;
        coordSystem.axisZ.vertices[1].position.y = 0;
        coordSystem.axisZ.vertices[1].position.z = 1 * length;
        coordSystem.axisZ.setColor(Color.blue);
        this.addCordSystem(coordSystem);
    }

    public class CoordSystem {

        public Line3D axisX = new Line3D(Color.red);
        public Line3D axisY = new Line3D(Color.green);
        public Line3D axisZ = new Line3D(Color.blue);

        private Vector3 temp = new Vector3();

        private void setAxisColor(Line3D axis, Color color) {
            axis.setColor(color);
        }

        public void setAxisXColor(Color color) {
            this.setAxisColor(axisX, color);
        }

        public void setAxisYColor(Color color) {
            this.setAxisColor(axisY, color);
        }

        public void setAxisZColor(Color color) {
            this.setAxisColor(axisZ, color);
        }

    }

    private Map<CoordSystem, CoordSystem3D> coords = new HashMap<CoordSystem, CoordSystem3D>();

    private static void findMinsMaxs(Vertex3D[] vecs, MinMaxs mms) {
        for (Vertex3D ve : vecs) {
            Vector3 v = ve.position;
            mms.minx = (mms.minx == null || v.x < mms.minx) ? v.x : mms.minx;
            mms.miny = (mms.miny == null || v.y < mms.miny) ? v.y : mms.miny;
            mms.minz = (mms.minz == null || v.z < mms.minz) ? v.z : mms.minz;
            mms.maxx = (mms.maxx == null || v.x > mms.maxx) ? v.x : mms.maxx;
            mms.maxy = (mms.maxy == null || v.y > mms.maxy) ? v.y : mms.maxy;
            mms.maxz = (mms.maxz == null || v.z > mms.maxz) ? v.z : mms.maxz;
        }
    }

    private static void findMinsMaxs(Object3D object3D, MinMaxs mms) {
        for (int fa = 0; fa < object3D.getQuads3DCount(); fa++) {
            Quad3D quad3D = object3D.getQuad3D(fa);
            findMinsMaxs(quad3D.vertices, mms);
        }
        for (int fa = 0; fa < object3D.getLines3DCount(); fa++) {
            Line3D line3D = object3D.getLine3D(fa);
            findMinsMaxs(line3D.vertices, mms);
        }
    }

    private List<Float> minxs = new ArrayList<Float>();
    private List<Float> maxxs = new ArrayList<Float>();
    private List<Float> minys = new ArrayList<Float>();
    private List<Float> maxys = new ArrayList<Float>();
    private List<Float> minzs = new ArrayList<Float>();
    private List<Float> maxzs = new ArrayList<Float>();

    class MinMaxs {

        Float minx = null;
        Float maxx = null;
        Float miny = null;
        Float maxy = null;
        Float minz = null;
        Float maxz = null;
    }

    private Map<Object3D, SceneParameters> sceneParameters = new HashMap<Object3D, SceneParameters>();

    @Override
    public void addObject3D(Object3D object3D) {
        SceneParameters sceneParameters = SceneParameters.getSceneParameters(object3D, new SceneParameters());
        minxs.add(sceneParameters.getX0());
        maxxs.add(sceneParameters.getX1());
        minys.add(sceneParameters.getY0());
        maxys.add(sceneParameters.getY1());
        minzs.add(sceneParameters.getZ0());
        maxzs.add(sceneParameters.getZ1());
        this.box3D.update();
        this.sceneParameters.put(object3D, sceneParameters);
        super.addObject3D(object3D);
    }

    @Override
    public void removeObject3D(Object3D object3D) {
        SceneParameters sceneParameters = this.sceneParameters.get(object3D);
        minxs.remove(sceneParameters.getX0());
        maxxs.remove(sceneParameters.getX1());
        minys.remove(sceneParameters.getY0());
        maxys.remove(sceneParameters.getY1());
        minzs.remove(sceneParameters.getZ0());
        maxzs.remove(sceneParameters.getZ1());
        this.sceneParameters.remove(object3D);
        this.box3D.update();
        super.removeObject3D(object3D);
    }

    @Override
    public void updateObject3DGeometry(Object3D object3D) {
        SceneParameters sceneParameters = this.sceneParameters.get(object3D);
        minxs.remove(sceneParameters.getX0());
        maxxs.remove(sceneParameters.getX1());
        minys.remove(sceneParameters.getY0());
        maxys.remove(sceneParameters.getY1());
        minzs.remove(sceneParameters.getZ0());
        maxzs.remove(sceneParameters.getZ1());
        sceneParameters = SceneParameters.getSceneParameters(object3D, sceneParameters);
        minxs.add(sceneParameters.getX0());
        maxxs.add(sceneParameters.getX1());
        minys.add(sceneParameters.getY0());
        maxys.add(sceneParameters.getY1());
        minzs.add(sceneParameters.getZ0());
        maxzs.add(sceneParameters.getZ1());
        this.box3D.update();
        super.updateObject3DGeometry(object3D);
    }

    public void addCordSystem(CoordSystem coordSystem) {
        CoordSystem3D c3D = new CoordSystem3D(coordSystem);
        this.coords.put(coordSystem, c3D);
        this.addObject3D(c3D);
    }

    public void removeCoordSystem(CoordSystem coordSystem) {
        CoordSystem3D c3D = this.coords.get(coordSystem);
        this.removeObject3D(c3D);
        this.coords.remove(coordSystem);
    }

    private boolean wasAdded = false;

    public void setBox(boolean isBox) {
        if (isBox) {
            if (this.wasAdded == false) {
                super.addObject3D(box3D);
                this.wasAdded = true;
            } else {
                super.updateObject3DGeometry(box3D);
            }
        } else {
            if (this.wasAdded) {
                this.wasAdded = false;
                super.removeObject3D(box3D);
            }
        }
    }
}
