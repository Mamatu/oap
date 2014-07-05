package ogla.legend;

import ogla.excore.Help;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.plugin.image.ImageComponent;

public class BundleLegend implements ImageComponentBundle {

    public BundleLegend() {
        //       root.documents.add(doc);
    }
//    private Help.Document doc = new DefaultDocumentImpl("overview", "default_view_help.html", BundleLegend.class);
    //  private DefaultChapterImpl root = new DefaultChapterImpl("Default view");
    private Help help = null;

    public ImageComponent newImageComponent() {
        Legend legend = new Legend(this);
        return legend;
    }

    public ImageComponent load(byte[] bytes) {
        return null;
    }

    public boolean canBeSaved() {
        return true;
    }

    public String getLabel() {
        return "Legend";
    }

    public String getSymbolicName() {
        return "analysis_legend_#@W@!#&FS*A";
    }

    public String getDescription() {
        return "";
    }

    public Help getHelp() {
        return help;
    }
}
