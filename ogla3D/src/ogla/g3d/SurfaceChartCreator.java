/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.awt.Color;
import java.util.List;
import ogla.math.Complex;
import ogla.math.MathStructure;
import ogla.math.ParserImpl;
import ogla.math.ParserCreator;
import ogla.math.SyntaxErrorException;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class SurfaceChartCreator {

    public interface Deepness {

        public float calculate(float x, float y);
    }

    public interface ChartPainter {

        public Color calculate(float x, float y, float z, SceneParameters sceneParameters);
    }

    public static Object3D updateSurfaceChart(Deepness deepness, ChartPainter painter, Object3D obj) {
        Object3D object3DImpl = obj;
        SceneParameters sp = SceneParameters.getSceneParameters(object3DImpl, new SceneParameters());
        for (int fa = 0; fa < object3DImpl.getQuads3DCount(); fa++) {
            Quad3D quad = object3DImpl.getQuad3D(fa);
            for (int fb = 0; fb < quad.vertices.length; fb++) {
                float x = quad.vertices[fb].position.x;
                float y = quad.vertices[fb].position.y;
                quad.vertices[fb].position.z = deepness.calculate(x, y);
                quad.vertices[fb].color = painter.calculate(x, y, quad.vertices[fb].position.z, sp);
            }
        }
        return object3DImpl;
    }

    public static Object3D updateSurfaceChart(SceneParameters sp, Deepness deepness, ChartPainter painter, Object3D obj) {
        Object3D object3DImpl = obj;
        for (int fa = 0; fa < object3DImpl.getQuads3DCount(); fa++) {
            Quad3D quad = object3DImpl.getQuad3D(fa);
            for (int fb = 0; fb < quad.vertices.length; fb++) {
                float x = quad.vertices[fb].position.x;
                float y = quad.vertices[fb].position.y;
                quad.vertices[fb].position.z = deepness.calculate(x, y);
                quad.vertices[fb].color = painter.calculate(x, y, quad.vertices[fb].position.z, sp);
            }
        }
        return object3DImpl;
    }

    public static Object3D createSurfaceChart(SceneParameters sceneParameters, Deepness deepness, ChartPainter painter, Object3D obj) {
        Object3D object3DImpl = obj;
        int quadsCounter = 0;
        int quadsCount = obj.getQuads3DCount();
        float x = sceneParameters.getX0();
        while (x < sceneParameters.getX1()) {
            float y = sceneParameters.getY0();
            while (y < sceneParameters.getY1()) {
                Quad3D quad = null;
                if (quadsCounter >= quadsCount) {
                    quad = new Quad3D();
                    object3DImpl.addQuad(quad);
                } else {
                    quad = obj.getQuad3D(quadsCounter);
                }
                quad.vertices[0].position.x = x;
                quad.vertices[0].position.y = y;
                quad.vertices[0].position.z = deepness.calculate(x, y);
                quad.vertices[0].color = painter.calculate(quad.vertices[0].position.x, quad.vertices[0].position.y, quad.vertices[0].position.z, sceneParameters);
                quad.vertices[1].position.x = x + sceneParameters.stepx;
                quad.vertices[1].position.y = y;
                quad.vertices[1].position.z = deepness.calculate(x + sceneParameters.stepx, y);
                quad.vertices[1].color = painter.calculate(quad.vertices[1].position.x, quad.vertices[1].position.y, quad.vertices[1].position.z, sceneParameters);
                quad.vertices[2].position.x = x + sceneParameters.stepx;
                quad.vertices[2].position.y = y + sceneParameters.stepy;
                quad.vertices[2].position.z = deepness.calculate(x + sceneParameters.stepx, y + sceneParameters.stepy);
                quad.vertices[2].color = painter.calculate(quad.vertices[2].position.x, quad.vertices[2].position.y, quad.vertices[2].position.z, sceneParameters);
                quad.vertices[3].position.x = x;
                quad.vertices[3].position.y = y + sceneParameters.stepy;
                quad.vertices[3].position.z = deepness.calculate(x, y + sceneParameters.stepy);
                quad.vertices[3].color = painter.calculate(quad.vertices[3].position.x, quad.vertices[3].position.y, quad.vertices[3].position.z, sceneParameters);
                quadsCounter++;
                y += sceneParameters.stepy;
            }
            x += sceneParameters.stepx;
        }
        return object3DImpl;
    }

    public static Object3D createSurfaceChart(SceneParameters sceneParameters, Deepness deepness, ChartPainter painter) {
        return createSurfaceChart(sceneParameters, deepness, painter, new Object3D());
    }

    private static Complex complex = new Complex();
    private static Complex complex1 = new Complex();

    private static float calculate(float x, float y, Object code, ParserImpl parser) throws SyntaxErrorException {
        complex.set(x);
        complex1.set(y);
        parser.setVariableValue("x", complex);
        parser.setVariableValue("y", complex1);
        MathStructure[] ms = parser.execute(code);
        Complex out = (Complex) ms[0];
        return out.re.floatValue();
    }

    public static Object createSurfaceChart(SceneParameters sceneParameters, String equation, ChartPainter chartPainter, Object3D obj, ParserImpl parser) throws SyntaxErrorException {
        Object3D object3DImpl = obj;
        int quadsCounter = 0;
        parser.setVariableValue("x", complex);
        parser.setVariableValue("y", complex1);
        Object code = parser.parse(equation);
        float stepx = sceneParameters.stepx;
        float stepy = sceneParameters.stepy;
        int quadsCount = obj.getQuads3DCount();
        float x = sceneParameters.getX0();
        while (x < sceneParameters.getX1()) {
            float y = sceneParameters.getY0();
            while (y < sceneParameters.getY1()) {
                Quad3D quad = null;
                if (quadsCounter >= quadsCount) {
                    quad = new Quad3D();
                    object3DImpl.addQuad(quad);
                } else {
                    quad = obj.getQuad3D(quadsCounter);
                }
                quad.vertices[0].position.x = x;
                quad.vertices[0].position.y = y;
                quad.vertices[0].position.z = calculate(x, y, code, parser);
                quad.vertices[0].color = chartPainter.calculate(quad.vertices[0].position.x,
                        quad.vertices[0].position.y, quad.vertices[0].position.z, sceneParameters);
                quad.vertices[1].position.x = x + stepx;
                quad.vertices[1].position.y = y;
                quad.vertices[1].position.z = calculate(x, y, code, parser);
                quad.vertices[1].color = chartPainter.calculate(quad.vertices[1].position.x,
                        quad.vertices[1].position.y, quad.vertices[1].position.z, sceneParameters);
                quad.vertices[2].position.x = x + stepx;
                quad.vertices[2].position.y = y + stepy;
                quad.vertices[2].position.z = calculate(x, y, code, parser);
                quad.vertices[2].color = chartPainter.calculate(quad.vertices[2].position.x,
                        quad.vertices[2].position.y, quad.vertices[2].position.z, sceneParameters);
                quad.vertices[3].position.x = x;
                quad.vertices[3].position.y = y + stepy;
                quad.vertices[3].position.z = calculate(x, y, code, parser);
                quad.vertices[3].color = chartPainter.calculate(quad.vertices[3].position.x,
                        quad.vertices[3].position.y, quad.vertices[3].position.z, sceneParameters);
                quadsCounter++;
                y += stepy;
            }
            x += stepx;
        }
        return code;
    }

    public static Object3D updateSurfaceChart(SceneParameters sceneParameters, ChartPainter painter,
            ParserImpl parser, Object code, Object3D obj) throws SyntaxErrorException {
        if ((code instanceof List) == false) {
        }
        parser.setVariableValue("x", complex);
        parser.setVariableValue("y", complex1);
        for (int fa = 0; fa < obj.getQuads3DCount(); fa++) {
            Quad3D quad = obj.getQuad3D(fa);
            for (int fb = 0; fb < quad.vertices.length; fb++) {
                float x = quad.vertices[fb].position.x;
                float y = quad.vertices[fb].position.y;
                quad.vertices[fb].position.z = calculate(x, y, code, parser);
                quad.vertices[fb].color = painter.calculate(x, y, quad.vertices[fb].position.z, sceneParameters);
            }
        }
        return obj;
    }

    public static Object3D updateSurfaceChart(SceneParameters sceneParameters, ChartPainter painter,
            ParserImpl parser, String equation, Object3D obj) throws SyntaxErrorException {
        parser.setVariableValue("x", complex);
        parser.setVariableValue("y", complex1);
        Object code = parser.parse(equation);
        for (int fa = 0; fa < obj.getQuads3DCount(); fa++) {
            Quad3D quad = obj.getQuad3D(fa);
            for (int fb = 0; fb < quad.vertices.length; fb++) {
                float x = quad.vertices[fb].position.x;
                float y = quad.vertices[fb].position.y;
                quad.vertices[fb].position.z = calculate(x, y, code, parser);
                quad.vertices[fb].color = painter.calculate(x, y, quad.vertices[fb].position.z, sceneParameters);
            }
        }
        return obj;
    }
}
