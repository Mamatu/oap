package ogla.chart;


import java.util.ArrayList;
import java.util.List;
import ogla.core.util.Reader;
import ogla.core.util.Writer;

public class IOManager {

    public List<IOEntity> iOEntities = new ArrayList<IOEntity>();

    public void load(Reader reader) {
        try {
            for (IOEntity iOEntity : iOEntities) {
                iOEntity.load(reader);
            }
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }

    public void save(Writer writer) {
        try {
            for (IOEntity iOEntity : iOEntities) {
                iOEntity.save(writer);
            }
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
    }
}
