
package ogla.axischart2D.popup;

import ogla.chart.gui.InternalFrame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JDesktopPane;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.event.InternalFrameEvent;
import javax.swing.event.InternalFrameListener;

/**
 *
 * @author marcin
 */
public class FramesPopupMenu extends JPopupMenu {

    InternalFrame frame1;
    InternalFrame frame2;
    InternalFrame frame3;
    InternalFrame frame4;
    InternalFrame frame5;
    JDesktopPane desktop;

    public FramesPopupMenu(JDesktopPane desktop, InternalFrame chartFrames, InternalFrame dataFrame,
            InternalFrame componentsFrame, InternalFrame propertiesFrame) {
        super();
        this.add(item1);
        this.add(item2);
        this.add(item3);
        this.add(item4);
        frame1 = chartFrames;
        frame2 = dataFrame;
        frame3 = componentsFrame;
        frame4 = propertiesFrame;
        ItemAction itemAction1 = new ItemAction(frame1, item1);
        ItemAction itemAction2 = new ItemAction(frame2, item2);
        ItemAction itemAction3 = new ItemAction(frame3, item3);
        ItemAction itemAction4 = new ItemAction(frame4, item4);

        item1.addActionListener(itemAction1);
        item2.addActionListener(itemAction2);
        item3.addActionListener(itemAction3);
        item4.addActionListener(itemAction4);

        frame1.addInternalFrameListener(itemAction1);
        frame2.addInternalFrameListener(itemAction2);
        frame3.addInternalFrameListener(itemAction3);
        frame4.addInternalFrameListener(itemAction4);

        this.desktop = desktop;
    }
    private JMenuItem item1 = new JMenuItem("Chart frame - refresh");
    private JMenuItem item2 = new JMenuItem("Data frame - refresh");
    private JMenuItem item3 = new JMenuItem("Components frame - refresh");
    private JMenuItem item4 = new JMenuItem("Properties frame - refresh");

    private class ItemAction implements ActionListener, InternalFrameListener {

        private InternalFrame frame;
        private JMenuItem item;

        public ItemAction(InternalFrame frame, JMenuItem item) {
            this.item = item;
            this.frame = frame;
        }

        public void actionPerformed(ActionEvent e) {
            item.setSelected(true);
            frame.setVisible(true);
            frame.setIcon(false);
            desktop.setComponentZOrder(frame, 0);
        }

        public void internalFrameOpened(InternalFrameEvent e) {
        }

        public void internalFrameClosing(InternalFrameEvent e) {
        }

        public void internalFrameClosed(InternalFrameEvent e) {
        }

        public void internalFrameIconified(InternalFrameEvent e) {
            item.setSelected(false);
        }

        public void internalFrameDeiconified(InternalFrameEvent e) {
            item.setSelected(true);
        }

        public void internalFrameActivated(InternalFrameEvent e) {
        }

        public void internalFrameDeactivated(InternalFrameEvent e) {
        }
    }
}
