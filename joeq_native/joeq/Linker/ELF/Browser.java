package Linker.ELF;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Iterator;
import java.util.List;

/**
 * @author John Whaley
 * @version $Id$
 */
public class Browser {

    public static void main(String[] args) throws IOException {
        RandomAccessFile f = new RandomAccessFile(args[0], "r");
        browseFile(f);
    }
    
    public static void browseFile(RandomAccessFile file) throws IOException {
        ELFRandomAccessFile f = new ELFRandomAccessFile(file);
        List sections = f.getSections();
        for (Iterator i=sections.iterator(); i.hasNext(); ) {
            Section s = (Section) i.next();
            System.out.println(s);
        }
    }
    
}
