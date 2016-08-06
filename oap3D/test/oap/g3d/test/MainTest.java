/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d.test;

import java.awt.Color;
import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.g3d.AxisChart3D;
import ogla.g3d.Camera;
import ogla.g3d.Line3D;
import ogla.g3d.Object3D;
import ogla.g3d.RendererVBO;
import ogla.g3d.SceneParameters;
import ogla.g3d.SurfaceChartCreator;
import ogla.math.Complex;
import ogla.math.ParserImpl;
import ogla.math.ParserCreator;
import ogla.math.SyntaxErrorException;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class MainTest {

    private static float gauss(float A, float x, float x0, float y, float y0, float sigmax, float sigmay) {
        float xx0 = (x - x0);
        float yy0 = (y - y0);
        float xx02 = xx0 * xx0;
        float yy02 = yy0 * yy0;
        xx02 = xx02 / (2 * sigmax * sigmax);
        yy02 = yy02 / (2 * sigmay * sigmay);
        return A * (float) Math.exp(-(xx02 + yy02));
    }

    private static String eq = "5*exp(-((x - 5)^2 + (y+0.05*t - 5)^2))";

    public static void main(String[] args) throws InterruptedException {
        TestFrame testFrame = new TestFrame();
        RendererVBO rv = new RendererVBO();
        Camera camera = new Camera();
        AxisChart3D oglaCanvas = new AxisChart3D(rv, camera, testFrame.getPanel());
        testFrame.setSize(800, 600);
        testFrame.setVisible(true);
        //rv.drawEdges(true);
        oglaCanvas.setBox(true);
        oglaCanvas.setDefaultCoordSystem(10.f);
        Vector3 vec = new Vector3(-20.f, -20.f, 20.f);
        Vector3 vec1 = new Vector3(0.f, 0.f, 0.f);
        camera.setCameraCoord(vec, vec1);
        oglaCanvas.setBackcolor(Color.GRAY);
        oglaCanvas.setBox(true);
        SceneParameters sceneParameters = new SceneParameters(0.f, 0.f, 0, 10.f, 10.f, 10.f, 0.25f, 0.25f);
        ParserImpl parser = ParserCreator.create();
        Object3D grid = new Object3D();
        Object code = null;

        for (int fa = 0; fa < 1000; fa++) {
            final int index = fa;
            parser.setVariableValue("t", new Complex(fa));
            if (fa == 0) {
                try {
                    code = SurfaceChartCreator.createSurfaceChart(sceneParameters, eq, new SurfaceChartCreator.ChartPainter() {

                        public Color calculate(float x, float y, float z, SceneParameters sceneParameters) {
                            Vector3 v = new Vector3(z / 5.f, 1.f / (z + 1), 0);
                            v.normalize();
                            Color color = new Color(v.x, v.y, v.z);
                            return color;
                        }
                    }, grid, parser);
                } catch (SyntaxErrorException ex) {
                    Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
                }
                oglaCanvas.addObject3D(grid);
            } else {
                try {
                    grid = SurfaceChartCreator.updateSurfaceChart(sceneParameters, new SurfaceChartCreator.ChartPainter() {

                        public Color calculate(float x, float y, float z, SceneParameters sceneParameters) {
                            Vector3 v = new Vector3(z / 5.f, 1.f / (z + 1.f), 0);
                            v.normalize();
                            Color color = new Color(v.x, v.y, v.z);
                            return color;
                        }
                    }, parser, code, grid);
                } catch (SyntaxErrorException ex) {
                    Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
                }
                oglaCanvas.updateObject3DGeometry(grid);
            }
            //  rv.rotate(new Quaternion(3.14f / 2.f, new Vector3(1, 1, 0)), object3D);
            oglaCanvas.repaint3D();
            System.out.print("|");
            //Thread.sleep(200);
        }
    }
}
