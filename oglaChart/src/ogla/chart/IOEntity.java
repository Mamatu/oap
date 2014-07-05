package ogla.chart;

import java.io.IOException;
import ogla.core.util.Reader;
import ogla.core.util.Writer;

public interface IOEntity {

    public void save(Writer writer) throws IOException;

    public void load(Reader reader) throws IOException, Reader.EndOfBufferException;
}
