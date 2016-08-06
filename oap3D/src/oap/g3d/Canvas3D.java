/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.awt.Color;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLJPanel;
import javax.media.opengl.glu.GLU;
import javax.swing.JPanel;
import ogla.math.Vector3;

public class Canvas3D extends GLJPanel implements GLEventListener {

    private GLU glu = null;
    private Renderer renderer = null;

    public Canvas3D(Renderer renderer, Camera camera) {
        super(createGLCapabilities());
        this.renderer = renderer;
        this.camera = camera;
    }

    public Canvas3D(Renderer renderer, Camera camera, JPanel jPanel) {
        super(createGLCapabilities());
        this.renderer = renderer;
        this.init(jPanel);
        this.camera = camera;
    }

    private void printStatus(GL gl) {
        String text = String.format("status = %d", gl.glGetError());
        System.err.println(text);
    }

    private static GLCapabilities createGLCapabilities() {
        GLCapabilities capabilities = new GLCapabilities();
        capabilities.setRedBits(8);
        capabilities.setBlueBits(8);
        capabilities.setGreenBits(8);
        capabilities.setAlphaBits(8);
        return capabilities;
    }

    private void init(final JPanel jPanel, int x, int y, int width, int height) {
        addGLEventListener(this);
        jPanel.add(this);
        this.setSize(width, height);
        this.setLocation(x, y);
        jPanel.addComponentListener(new ComponentListener() {

            public void componentResized(ComponentEvent e) {
                Canvas3D.this.setSize(jPanel.getSize());
            }

            public void componentMoved(ComponentEvent e) {
            }

            public void componentShown(ComponentEvent e) {
            }

            public void componentHidden(ComponentEvent e) {
            }
        });
    }

    private void init(final JPanel jPanel) {
        this.init(jPanel, 0, 0, jPanel.getWidth(), jPanel.getHeight());
    }

    public void init(GLAutoDrawable drawable) {
        final GL gl = drawable.getGL();
        gl.glEnable(GL.GL_DEPTH_TEST);
        gl.glDepthFunc(GL.GL_LEQUAL);
        gl.glShadeModel(GL.GL_SMOOTH_LINE_WIDTH_RANGE);
        gl.glClearColor(this.backcolor.x, this.backcolor.y, this.backcolor.z, 0f);
        gl.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, GL.GL_NICEST);
        this.glu = new GLU();
    }
    
    public interface Component3D {

        public void display(GL gl, GLU glu);
    }

    private List<Component3D> component3Ds = new ArrayList<Component3D>();

    public void addComponent3D(Component3D component3D) {
        this.component3Ds.add(component3D);
    }

    public void removeComponent3D(Component3D component3D) {
        this.component3Ds.remove(component3D);
    }

    private boolean isDestroyed = false;

    public void display(GLAutoDrawable drawable) {
        final GL gl = drawable.getGL();
        synchronized (this.renderer) {
            gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
            gl.glClearColor(this.backcolor.x, this.backcolor.y, this.backcolor.z, 0f);
            setCamera(gl, glu);
            this.renderer.render(gl, glu);
            for (Component3D component3D : this.component3Ds) {
                component3D.display(gl, glu);
            }
        }
    }

    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
        final GL gl = drawable.getGL();
        gl.glViewport(0, 0, width, height);
    }

    /**
     * Changing devices is not supported.
     *
     * @see
     * javax.media.opengl.GLEventListener#displayChanged(javax.media.opengl.GLAutoDrawable,
     * boolean, boolean)
     */
    public void displayChanged(GLAutoDrawable drawable, boolean modeChanged, boolean deviceChanged) {
    }

    private void setCamera(GL gl, GLU glu) {
        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glLoadIdentity();
        float widthHeightRatio = (float) this.getWidth() / (float) this.getHeight();
        glu.gluPerspective(45, widthHeightRatio, 1, 1000);
        glu.gluLookAt(camera.position.x, camera.position.y, camera.position.z,
                camera.lookAt.x, camera.lookAt.y, camera.lookAt.z,
                camera.orientation.x, camera.orientation.y, camera.orientation.z);
        gl.glMatrixMode(GL.GL_MODELVIEW);
        gl.glLoadIdentity();
    }

    public void dispose(GLAutoDrawable drawable) {
    }

    public void repaint3D() {
        this.display();
    }

    public void addObject3D(Object3D object3D) {
        synchronized (this.renderer) {
            renderer.addObject3D(object3D);
        }
    }

    public void removeObject3D(Object3D object3D) {
        synchronized (this.renderer) {
            renderer.removeObject3D(object3D);
        }
    }

    public void updateObject3DGeometry(Object3D object3D) {
        synchronized (this.renderer) {
            renderer.updateObject3DGeometry(object3D);
        }
    }

    private Camera camera = null;

    private Vector3 backcolor = new Vector3();

    public void setBackcolor(Vector3 color) {
        backcolor.set(color);
    }

    public void setBackcolor(Color color) {
        Vector3 vec = new Vector3(color.getRed(), color.getGreen(), color.getBlue());
        this.setBackcolor(vec);
    }
}
