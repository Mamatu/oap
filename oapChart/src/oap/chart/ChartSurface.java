package ogla.chart;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.List;

/**
 * Class which manage SurfaceElement's objects.
 */
public class ChartSurface extends javax.swing.JPanel {

    private Rectangle resolution = new Rectangle();

    public void fullscreen() {
        resolution = this.getBounds(resolution);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        this.setBounds(0, 0, screenSize.width, screenSize.height);
    }

    public void returnToPreviousResolution() {
        this.setBounds(resolution);
    }

    public ChartSurface() {
        resolution = this.getBounds(resolution);
        setVisible(true);
        setPreferredSize(new Dimension(250, 250));
        setSize(new Dimension(250, 250));
        setDoubleBuffered(true);
        this.addKeyListener(new KeyListener() {

            public void keyTyped(KeyEvent e) {
            }

            public void keyPressed(KeyEvent e) {
            }

            public void keyReleased(KeyEvent e) {
                if (KeyEvent.VK_F5 == e.getKeyCode()) {
                    ChartSurface.this.fullscreen();
                } else if (KeyEvent.VK_ESCAPE == e.getKeyCode()) {
                    ChartSurface.this.returnToPreviousResolution();
                }
            }
        });
        repaint();
    }
    protected List<DrawableOnChartSurface> elements = new ArrayList<DrawableOnChartSurface>();

    /**
     * Return objects which are drawable on ChartSurface.
     * @return
     */
    public List<DrawableOnChartSurface> getDrawable() {
        return elements;
    }

    @Override
    public void paintComponent(java.awt.Graphics g) {
        super.paintComponent(g);
        Graphics2D gd = (Graphics2D) g;
        this.draw(gd);
    }

    public void setDrawingType(DrawingInfo drawingInfo) {
        this.drawingInfo = drawingInfo;
    }
    private DrawingInfo drawingInfo = DrawingInfo.DrawingOnChart;

    /**
     * Special method which storage information about how chart should be drawn. If you want to draw chart in another way,
     * you should override this method.
     * @param gd
     */
    public void draw(Graphics2D gd) {
        Color backColor = this.getBackground();
        gd.setBackground(backColor);
        gd.clearRect(0, 0, this.getSize().width, this.getSize().height);
        for (DrawableOnChartSurface se : getDrawable()) {
            se.drawOnChartSurface(gd, this.drawingInfo);
        }
        this.drawingInfo = DrawingInfo.DrawingOnChart;
    }

    protected void initComponents() {
        this.addComponentListener(new java.awt.event.ComponentAdapter() {

            @Override
            public void componentResized(java.awt.event.ComponentEvent evt) {
                ChartSurface.this.repaint();
            }
        });
    }
}
