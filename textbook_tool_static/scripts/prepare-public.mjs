import { cp, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const publicDir = join(root, "public");

await mkdir(publicDir, { recursive: true });
await cp(join(root, "index.html"), join(publicDir, "index.html"), { force: true });
await cp(join(root, "src"), join(publicDir, "src"), { recursive: true, force: true });
await cp(join(root, "nuclei_library_stats"), join(publicDir, "nuclei_library_stats"), { recursive: true, force: true });

console.log("Prepared static public output.");
