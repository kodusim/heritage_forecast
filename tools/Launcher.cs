using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

internal static class Launcher {
    [STAThread]
    static int Main(string[] args) {
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string payload = Path.Combine(root, "_internal");
            if (!Directory.Exists(payload)) payload = Path.Combine(root, "portable", "HeritageStudio", "_internal");
            if (!Directory.Exists(payload)) payload = Path.Combine(root, "portable-v3", "HeritageStudio", "_internal");
            if (!Directory.Exists(payload)) payload = Path.Combine(root, "portable-v2", "HeritageStudio", "_internal");
            string runtime = Path.Combine(payload, "runtime", "pythonw.exe");
            string script = Path.Combine(payload, "app", "main.py");
            if (!File.Exists(runtime) || !File.Exists(script)) {
                MessageBox.Show("실행에 필요한 _internal 폴더를 찾지 못했습니다.\nZIP 전체를 압축 해제한 뒤 실행하세요. EXE만 따로 옮기지 마세요.\n바탕화면에서는 EXE의 바로가기를 만들어 사용하세요.", "Heritage Studio");
                return 2;
            }
            var command = new StringBuilder(Quote(script));
            foreach (string arg in args) command.Append(" ").Append(Quote(arg));
            var info = new ProcessStartInfo(runtime, command.ToString());
            info.WorkingDirectory = payload;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            using (var process = Process.Start(info)) { process.WaitForExit(); return process.ExitCode; }
        } catch (Exception ex) {
            MessageBox.Show("프로그램을 실행하지 못했습니다.\n" + ex.Message, "Heritage Studio");
            return 1;
        }
    }
    static string Quote(string value) {
        var s = new StringBuilder("\""); int slashes = 0;
        foreach (char c in value) {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { s.Append('\\', slashes * 2 + 1).Append(c); slashes = 0; continue; }
            s.Append('\\', slashes).Append(c); slashes = 0;
        }
        s.Append('\\', slashes * 2).Append('"'); return s.ToString();
    }
}
