/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.media.opengl.GL;
import javax.media.opengl.glu.GLU;
import ogla.core.util.Writer;
import ogla.g3d.utils.Tools3D;
import ogla.math.Quaternion;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class RendererVBO implements Renderer {

    private List<Object3DBundle> objects = new ArrayList<Object3DBundle>();
    private List<Object3DBundle> vboObjects = new ArrayList<Object3DBundle>();
    private List<Object3DBundle> msutBeRefreshedObjects = new ArrayList<Object3DBundle>();
    private Map<Object3D, Object3DBundle> object3DBundles = new HashMap<Object3D, Object3DBundle>();

    private class Object3DBundle {

        private Object3D object3D = null;
        Vector3 scale = null;
        Vector3 translate = null;
        Quaternion rotate = null;
        Integer[] ids = new Integer[4];
        boolean isCreatedVBO = false;
        private boolean drawEdges = false;
        private boolean drawFill = true;

        public void drawEdges(boolean draw) {
            this.drawEdges = draw;
        }

        public Object3DBundle(Object3D object3D) {
            this.object3D = object3D;
            this.quadsVertexWriter = new Writer(this.getQuadsBufferSize());
            this.quadsColorWriter = new Writer(this.getQuadsColorBufferSize());
            this.linesVertexWriter = new Writer(this.getLinesBufferSize());
            this.linesColorWriter = new Writer(this.getLinesColorBufferSize());
        }

        private int getQuadsBufferSize() {
            return object3D.getQuads3DCount() * 48;
        }

        private int getQuadsColorBufferSize() {
            return object3D.getQuads3DCount() * 48;
        }

        private int getLinesBufferSize() {
            return object3D.getLines3DCount() * 24;
        }

        private int getLinesColorBufferSize() {
            return object3D.getLines3DCount() * 24;
        }

        private void createVBOBuffer(GL gl, Integer id, Writer writer) {
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, id);
            ByteBuffer byteBuffer = ByteBuffer.wrap(writer.getBytes());
            gl.glBufferData(GL.GL_ARRAY_BUFFER, byteBuffer.capacity(), byteBuffer, GL.GL_DYNAMIC_DRAW);
            writer.clear();
        }

        private void createVBO(GL gl) {
            if (object3D.getQuads3DCount() > 0) {
                IntBuffer intBuffer = IntBuffer.allocate(2);
                gl.glGenBuffers(2, intBuffer);
                ids[0] = intBuffer.get();
                ids[1] = intBuffer.get();
                this.isCreatedVBO = true;
                Tools3D.quadsToBytes(object3D, this.quadsVertexWriter, this.quadsColorWriter);
                this.createVBOBuffer(gl, ids[0], this.quadsVertexWriter);
                this.createVBOBuffer(gl, ids[1], this.quadsColorWriter);
            }
            if (object3D.getLines3DCount() > 0) {
                IntBuffer intBuffer = IntBuffer.allocate(2);
                gl.glGenBuffers(2, intBuffer);
                ids[2] = intBuffer.get();
                ids[3] = intBuffer.get();
                Tools3D.linesToBytes(object3D, this.linesVertexWriter, this.linesColorWriter);
                this.createVBOBuffer(gl, ids[2], this.linesVertexWriter);
                this.createVBOBuffer(gl, ids[3], this.linesColorWriter);
            }
            if (object3D.getQuads3DCount() > 0 || object3D.getLines3DCount() > 0) {
                this.isCreatedVBO = true;
            }
        }

        private void destroyVBOBuffer(GL gl, Integer id) {
            synchronized (this) {
                IntBuffer intBuffer = IntBuffer.allocate(1);
                intBuffer.put(id);
                gl.glDeleteBuffers(1, intBuffer);
            }
        }

        private void destroyVBO(GL gl) {
            synchronized (this) {
                if (this.object3D.getQuads3DCount() > 0) {
                    this.destroyVBOBuffer(gl, ids[0]);
                    this.destroyVBOBuffer(gl, ids[1]);
                }
                if (this.object3D.getLines3DCount() > 0) {
                    this.destroyVBOBuffer(gl, ids[2]);
                    this.destroyVBOBuffer(gl, ids[3]);
                }
            }
        }

        private void displayVBO(GL gl, Integer id, Integer id1, int type, int elemtnsCount) {
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, id);
            gl.glEnableClientState(GL.GL_VERTEX_ARRAY);
            gl.glVertexPointer(3, GL.GL_FLOAT, 0, 0);
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, id1);
            gl.glEnableClientState(GL.GL_COLOR_ARRAY);
            gl.glColorPointer(4, GL.GL_FLOAT, 0, 0);
            gl.glDrawArrays(type, 0, elemtnsCount);
            gl.glDisableClientState(GL.GL_COLOR_ARRAY);
            gl.glDisableClientState(GL.GL_VERTEX_ARRAY);
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, 0);
        }

        private void drawVBOQuads(GL gl) {
            Integer id = this.ids[0];
            Integer id1 = this.ids[1];
            if (this.drawFill == true) {
                this.displayVBO(gl, id, id1, GL.GL_QUADS, this.object3D.getQuads3DCount() * 4);
            }
            if (this.drawEdges == true) {
                gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE);
                this.displayVBO(gl, id, id1, GL.GL_QUADS, this.object3D.getQuads3DCount() * 4);
                gl.glPolygonMode(GL.GL_FRONT, GL.GL_FILL);
            }
        }

        private void drawVBOLines(GL gl) {
            Integer id = this.ids[2];
            Integer id1 = this.ids[3];
            this.displayVBO(gl, id, id1, GL.GL_LINES, this.object3D.getLines3DCount() * 2);
        }

        private void drawVBO(GL gl) {
            synchronized (this) {
                gl.glPushMatrix();
                gl.glLoadIdentity();
                if (this.scale != null) {
                    Vector3 vec = this.scale;
                    gl.glScalef(vec.x, vec.y, vec.z);
                    this.scale = null;
                }
                if (this.translate != null) {
                    Vector3 vec = this.translate;
                    gl.glTranslatef(vec.x, vec.y, vec.z);
                    this.translate = null;
                }
                if (this.rotate != null) {
                    Quaternion q = this.rotate;
                    gl.glRotatef(q.w, q.xyz.x, q.xyz.y, q.xyz.z);
                    this.rotate = null;
                }
                if (object3D.getQuads3DCount() > 0) {
                    this.drawVBOQuads(gl);
                }
                if (object3D.getLines3DCount() > 0) {
                    this.drawVBOLines(gl);
                }
                gl.glPopMatrix();
            }
        }

        private void updateVBOBuffer(GL gl, Integer id, Writer writer) {
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, id);
            ByteBuffer byteBuffer = gl.glMapBuffer(GL.GL_ARRAY_BUFFER, GL.GL_READ_WRITE);
            if (byteBuffer != null) {
                byteBuffer.put(writer.getBytes());
                gl.glUnmapBuffer(GL.GL_ARRAY_BUFFER);
                writer.clear();
            }
        }

        private void updateVBO(GL gl) {
            synchronized (this) {
                if (this.isCreatedVBO == true) {
                    if (object3D.getQuads3DCount() > 0) {
                        Integer id = ids[0];
                        Integer id1 = ids[1];
                        Writer writer = this.quadsVertexWriter;
                        Writer writer1 = this.quadsColorWriter;
                        Tools3D.quadsToBytes(object3D, writer, writer1);
                        this.updateVBOBuffer(gl, id, writer);
                        this.updateVBOBuffer(gl, id1, writer1);
                    }

                    if (object3D.getLines3DCount() > 0) {
                        Integer id = ids[2];
                        Integer id1 = ids[3];
                        Writer writer = this.linesVertexWriter;
                        Writer writer1 = this.linesColorWriter;
                        Tools3D.linesToBytes(object3D, writer, writer1);
                        this.updateVBOBuffer(gl, id, writer);
                        this.updateVBOBuffer(gl, id1, writer1);
                    }
                } else {
                    this.createVBO(gl);
                }
            }
        }

        private Writer quadsVertexWriter = null;
        private Writer quadsColorWriter = null;
        private Writer linesVertexWriter = null;
        private Writer linesColorWriter = null;

    }

    public void addObject3D(Object3D object3D) {
        Object3DBundle bundle = new Object3DBundle(object3D);
        this.object3DBundles.put(object3D, bundle);
        this.objects.add(bundle);
    }

    public void removeObject3D(Object3D object3D) {
        Object3DBundle bundle = this.object3DBundles.get(object3D);
        this.objects.remove(object3D);
    }

    public void updateObject3DGeometry(Object3D object3D) {
        Object3DBundle bundle = this.object3DBundles.get(object3D);
        msutBeRefreshedObjects.add(bundle);
    }

    public void render(GL gl, GLU glu) {
        synchronized (this) {
            for (Object3DBundle object3D : this.msutBeRefreshedObjects) {
                object3D.updateVBO(gl);
            }
            for (Object3DBundle object3D : vboObjects) {
                if (this.objects.contains(object3D) == false) {
                    object3D.destroyVBO(gl);
                    this.vboObjects.remove(object3D);
                }
            }
            for (int fa = 0; fa < objects.size(); fa++) {
                Object3DBundle object3D = objects.get(fa);
                if (vboObjects.contains(objects.get(fa)) == false) {
                    object3D.createVBO(gl);
                    this.vboObjects.add(objects.get(fa));
                }
                object3D.drawVBO(gl);
            }
        }
    }

    public void scale(Vector3 vec, Object3D object3D) {
        Object3DBundle bundle = this.object3DBundles.get(object3D);
        bundle.scale = new Vector3(vec);
    }

    public void translate(Vector3 vec, Object3D object3D) {
        Object3DBundle bundle = this.object3DBundles.get(object3D);
        bundle.translate = new Vector3(vec);
    }

    public void rotate(Quaternion quaternion, Object3D object3D) {
        Object3DBundle bundle = this.object3DBundles.get(object3D);
        bundle.rotate = new Quaternion(quaternion);
    }
}
