package ogla.label;

import ogla.excore.Help;
import ogla.excore.util.Reader;
import ogla.excore.util.Reader.EndOfBufferException;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.plugin.image.ImageComponent;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class BundleLabel implements ImageComponentBundle {

    public Help getHelp() {
        return null;
    }

    public ImageComponent newImageComponent() {
        return new Label(this);
    }

    public ImageComponent load(byte[] bytes) {
        Label label = new Label(this);
        try {
            Reader reader = new Reader(bytes);

            int size = reader.readInt();
            String str = reader.readStr(size);
            label.setTop(str);

            size = reader.readInt();
            str = reader.readStr(size);
            label.setBottom(str);

            size = reader.readInt();
            str = reader.readStr(size);
            label.setRight(str);

            size = reader.readInt();
            str = reader.readStr(size);
            label.setLeft(str);
        } catch (EndOfBufferException ex) {
            Logger.getLogger(BundleLabel.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(BundleLabel.class.getName()).log(Level.SEVERE, null, ex);
        }
        return label;
    }

    public String getLabel() {
        return "Label";
    }

    public String getSymbolicName() {
        return "analysis_label_$f#f@a";
    }

    public boolean canBeSaved() {
        return true;
    }

    public String getDescription() {
        return "";
    }

    public boolean canBeAttachedTo(Class clazz) {
        return true;
    }
}
