// AppletIO.java, created Oct 5, 2004 8:40:10 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.gui;

import java.util.ArrayList;
import java.util.List;
import java.awt.Dimension;
import java.awt.Insets;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import javax.swing.JApplet;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;
import jwutil.io.FillableReader;
import jwutil.io.ReaderInputStream;
import jwutil.reflect.Reflect;

/**
 * AppletIO
 * 
 * @author jwhaley
 * @version $Id$
 */
public class AppletIO extends JApplet {
    
    public static String DEFAULT_ENCODING = "UTF-8";
    
    /**
     * AppletWriter takes anything written to it and puts it in the output area.
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
            outputArea.append(str);
            jumpToEndOfOutput();
        }

        /* (non-Javadoc)
         * @see java.io.Flushable#flush()
         */
        public void flush() throws IOException {
            // Nothing to do.
        }

        /* (non-Javadoc)
         * @see java.io.Closeable#close()
         */
        public void close() throws IOException {
            // Nothing to do.
        }
    }
    
    /**
     * AppletOutputStream takes anything written to it and puts it in the output
     * area.
     * 
     * TODO: The write() method doesn't handle multibyte characters correctly.
     * If you want correct usage of multibyte characters, use AppletWriter
     * instead.
     * 
     * @author jwhaley
     * @version $Id$
     */
    public class AppletOutputStream extends OutputStream {
        
        /* (non-Javadoc)
         * @see java.io.OutputStream#write(int)
         */
        public void write(int b) throws IOException {
            outputArea.append(new String(new byte[]{(byte) b}));
            jumpToEndOfOutput();
        }

        /* (non-Javadoc)
         * @see java.io.OutputStream#write(byte[], int, int)
         */
        public void write(byte b[], int off, int len) throws IOException {
            if (len == 0) {
                return;
            }
            String str = new String(b, off, len, DEFAULT_ENCODING);
            outputArea.append(str);
            jumpToEndOfOutput();
        }
    }
    
    /**
     * Listens for newline inputs in the input area, and sends that line
     * to inputWriter.
     * 
     * @author jwhaley
     * @version $Id$
     */
    public class TextAreaListener implements DocumentListener {
    
        /* (non-Javadoc)
         * @see javax.swing.event.DocumentListener#insertUpdate(javax.swing.event.DocumentEvent)
         */
        public void insertUpdate(DocumentEvent e) {
            int length = e.getLength();
            if (length == 0) return;
            int offset = e.getOffset();
            try {
                String s = inputArea.getText(offset, length);
                int i;
                while ((i = s.indexOf('\n')) >= 0) {
                    int lineNum = inputArea.getLineOfOffset(offset);
                    int sOff = inputArea.getLineStartOffset(lineNum);
                    int eOff = inputArea.getLineEndOffset(lineNum);
                    String line = inputArea.getText(sOff, eOff - sOff);
                    if (inputWriter != null) {
                        inputWriter.write(line);
                    }
                    offset = eOff;
                    s = s.substring(i + 1);
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
 
    /**
     * Scroll to the end of the output area.
     */
    public void jumpToEndOfOutput() {
        outputArea.setCaretPosition(outputArea.getDocument().getLength());
    }
    
    JTextArea outputArea;
    JTextArea inputArea;
    Writer inputWriter;
    Method method;
    Object[] methodArgs;

    protected void loadAppletParameters() {
        // Get the applet parameters.
        String className = getParameter("class");
        String methodName = getParameter("method");
        if (methodName == null) methodName = "main";
        method = Reflect.getDeclaredMethod(className, methodName);
        List mArgs = new ArrayList();
        for (int i = 0; ; ++i) {
            String arg = getParameter("arg"+i);
            if (arg == null) break;
            mArgs.add(arg);
        }
        methodArgs = mArgs.toArray();
    }
    
    public void init() {
        loadAppletParameters();
        
        //Execute a job on the event-dispatching thread:
        //creating this applet's GUI.
        try {
            javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
                public void run() {
                    createGUI();
                }
            });
        } catch (Exception e) { 
            System.err.println("createGUI didn't successfully complete");
        }
    }
    
    void createGUI() {
        outputArea = new JTextArea();
        outputArea.setMargin(new Insets(5, 5, 5, 5));
        outputArea.setEditable(false);
        inputArea = new JTextArea();
        inputArea.setMargin(new Insets(5, 5, 5, 5));
        JScrollPane top = new JScrollPane(outputArea);
        JScrollPane bottom = new JScrollPane(inputArea,
            JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
            JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, top, bottom);
        //splitPane.setDividerLocation(350);
        splitPane.setResizeWeight(1.0);
        getContentPane().add(splitPane);
        // Provide minimum sizes for the two components in the split pane.
        Dimension minimumSize = new Dimension(400, 40);
        top.setMinimumSize(minimumSize);
        bottom.setMinimumSize(minimumSize);
        bottom.setPreferredSize(minimumSize);
        // Use this listener to listen to changes to the text area.
        DocumentListener myListener = new TextAreaListener();
        inputArea.getDocument().addDocumentListener(myListener);
        // Redirect System.out/System.err to our text area.
        try {
            PrintStream out = new PrintStream(new AppletOutputStream(), true, DEFAULT_ENCODING);
            try {
                System.setOut(out);
                System.setErr(out);
                // Redirect System.in from our input area.
                FillableReader in = new FillableReader();
                inputWriter = in.getWriter();
                System.setIn(new ReaderInputStream(in));
            } catch (SecurityException x) {
                outputArea.append("Cannot reset stdio: " + x);
                x.printStackTrace(out);
            }
        } catch (UnsupportedEncodingException x) {
            outputArea.append(x.toString());
            x.printStackTrace();
        }
    }
    
    public static void main(String[] s) throws SecurityException,
        NoSuchMethodException, ClassNotFoundException {
        AppletIO applet = new AppletIO();
        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.getContentPane().add(applet);
        f.setSize(500, 400);
        f.setLocation(200, 200);
        applet.createGUI();
        f.setVisible(true);
        
        if (s.length < 1) {
            applet.method = AppletIO.class.getDeclaredMethod("example", new Class[0]);
            applet.methodArgs = new Object[0];
        } else {
            applet.method = Class.forName(s[0]).getDeclaredMethod("main",
                new Class[]{String[].class});
            String[] s2 = new String[s.length - 1];
            System.arraycopy(s, 1, s2, 0, s2.length);
            applet.methodArgs = new Object[]{s2};
        }
        System.out.println("Starting " + applet.method.getDeclaringClass().getSimpleName()+
            "."+applet.method.getName()+"()");
        launch(applet.method, applet.methodArgs);
    }
    
    public static void launch(final Method m, final Object[] args) {
        new Thread() {
            public void run() {
                try {
                    m.invoke(null, args);
                } catch (InvocationTargetException e) {
                    e.getTargetException().printStackTrace();
                } catch (IllegalArgumentException e) {
                    e.printStackTrace();
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }.start();
    }
    
    // For an example: A method that just consumes System.in.
    public static void example() {
        try {
            BufferedReader in = new BufferedReader(new InputStreamReader(
                System.in, DEFAULT_ENCODING));
            for (;;) {
                String s = in.readLine();
                System.out.println("IN: " + s);
            }
        } catch (IOException x) {
            x.printStackTrace();
        }
    }
}
