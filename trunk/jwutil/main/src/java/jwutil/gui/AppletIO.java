// AppletIO.java, created Oct 5, 2004 8:40:10 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.gui;

import java.awt.BorderLayout;
import java.awt.Insets;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Writer;
import javax.swing.JApplet;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;
import jwutil.io.FillableInputStream;

/**
 * AppletIO
 * 
 * @author jwhaley
 * @version $Id$
 */
public class AppletIO extends JApplet {
    
    /**
     * AppletWriter takes anything written to it and puts it in the text area.
     * 
     * @author jwhaley
     * @version $Id$
     */
    public class AppletWriter extends Writer {

        /* (non-Javadoc)
         * @see java.io.Writer#write(char[], int, int)
         */
        public void write(char[] cbuf, int off, int len) throws IOException {
            if (len == 0) {
                return;
            }
            
            String str = new String(cbuf, off, len);
            textArea.append(str);
        }

        /* (non-Javadoc)
         * @see java.io.Writer#flush()
         */
        public void flush() throws IOException {
            // Nothing to do.
        }

        /* (non-Javadoc)
         * @see java.io.Writer#close()
         */
        public void close() throws IOException {
            // Nothing to do.
        }
        
    }
    
    /**
     * AppletOutputStream takes anything written to it and puts it in the text area.
     * 
     * TODO: The write() method doesn't handle multibyte characters correctly.
     * If you want correct usage of multibyte characters, use AppletWriter instead.
     * 
     * @author jwhaley
     * @version $Id$
     */
    public class AppletOutputStream extends OutputStream {

        /* (non-Javadoc)
         * @see java.io.OutputStream#write(int)
         */
        public void write(int b) throws IOException {
            textArea.append(new String(new byte[] { (byte) b }));
        }
        
        /* (non-Javadoc)
         * @see java.io.OutputStream#write(byte[], int, int)
         */
        public void write(byte b[], int off, int len) throws IOException {
            if (len == 0) {
                return;
            }
            
            String str = new String(b, off, len);
            textArea.append(str);
        }
    }
    
    public class TextAreaListener implements DocumentListener {

        /* (non-Javadoc)
         * @see javax.swing.event.DocumentListener#insertUpdate(javax.swing.event.DocumentEvent)
         */
        public void insertUpdate(DocumentEvent e) {
            int length = e.getLength();
            if (length == 0) return;
            int offset = e.getOffset();
            try {
                synchronized (AppletIO.this) {
                    String s = inputArea.getText(offset, length);
                    int i;
                    while ((i = s.indexOf('\n')) >= 0) {
                        int lineNum = inputArea.getLineOfOffset(offset);
                        int sOff = inputArea.getLineStartOffset(lineNum);
                        int eOff = inputArea.getLineEndOffset(lineNum);
                        String line = inputArea.getText(sOff, eOff-sOff);
                        if (inputWriter != null) {
                            inputWriter.write(line);
                        }
                        offset = eOff;
                        s = s.substring(i+1);
                    }
                }
            } catch (IOException x) {
                x.printStackTrace();
            } catch (BadLocationException x) {
                x.printStackTrace();
            }
        }

        /* (non-Javadoc)
         * @see javax.swing.event.DocumentListener#removeUpdate(javax.swing.event.DocumentEvent)
         */
        public void removeUpdate(DocumentEvent e) {
            // Ignore.
        }

        /* (non-Javadoc)
         * @see javax.swing.event.DocumentListener#changedUpdate(javax.swing.event.DocumentEvent)
         */
        public void changedUpdate(DocumentEvent e) {
            // Ignore.
        }

    }
    
    JTextArea textArea;
    JTextArea inputArea;
    Writer inputWriter;
    
    public void init() {
        textArea = new JTextArea();
        textArea.setMargin(new Insets(5, 5, 5, 5));
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(new JScrollPane(textArea));
        textArea.setEditable(false);
        
        inputArea = new JTextArea();
        inputArea.setMargin(new Insets(5, 5, 5, 5));
        getContentPane().add(
            new JScrollPane(inputArea,
                JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER),
            BorderLayout.SOUTH);
        
        // Use this listener to listen to changes to the text area.
        DocumentListener myListener = new TextAreaListener();
        inputArea.getDocument().addDocumentListener(myListener);
        
        // Redirect System.out/System.err to our text area.
        PrintStream out = new PrintStream(new AppletOutputStream());
        System.setOut(out);
        System.setErr(out);
        
        // Redirect System.in from our input area.
        FillableInputStream in = new FillableInputStream();
        inputWriter = in.getWriter();
        System.setIn(in);
    }

    public static void main(String[] args) {
        JApplet applet = new AppletIO();
        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.getContentPane().add(applet);
        f.setSize(500, 400);
        f.setLocation(200, 200);
        applet.init();
        f.setVisible(true);
        
        System.out.println("Applet started.");

        new Thread() {
            public void run() {
                DataInputStream in = new DataInputStream(System.in);
                for (;;) {
                    try {
                        String s = in.readLine();
                        System.out.println("IN: "+s);
                    } catch (IOException x) {
                        x.printStackTrace();
                    }
                }
            }
        }.start();
    }
}
