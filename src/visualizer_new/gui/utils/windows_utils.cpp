/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/events.hpp"
#include "core_new/logger.hpp"

#include "gui/utils/windows_utils.hpp"

#ifdef WIN32
#include <ShlObj.h>
#include <windows.h>
#endif // WIN32

namespace lfs::vis::gui {
#ifdef WIN32

    namespace utils {
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[],
                                 UINT cFileTypes,
                                 bool blnDirectory) {

            HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(hr)) {
                LOG_ERROR("Failed to initialize COM: {:#x}", static_cast<unsigned int>(hr));
            } else {
                // Create the FileOpenDialog instance
                IFileOpenDialog* pFileOpen = nullptr;
                hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                                      IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

                if (SUCCEEDED(hr)) {
                    DWORD dwOptions;

                    if (SUCCEEDED(pFileOpen->GetOptions(&dwOptions))) {
                        if (blnDirectory) {
                            pFileOpen->SetOptions(dwOptions | FOS_PICKFOLDERS);
                        } else {
                            if (rgSpec != nullptr && cFileTypes > 0) {
                                hr = pFileOpen->SetFileTypes(cFileTypes, rgSpec);
                                if (SUCCEEDED(hr)) {
                                    pFileOpen->SetOptions(dwOptions | FOS_NOCHANGEDIR | FOS_FILEMUSTEXIST);
                                    pFileOpen->SetFileTypeIndex(1);
                                } else {
                                    LOG_ERROR("Failed to set file types: {:#x}", static_cast<unsigned int>(hr));
                                }
                            } else {
                                pFileOpen->SetOptions(dwOptions | FOS_NOCHANGEDIR | FOS_FILEMUSTEXIST);
                            }
                        }
                    }

                    // Show the Open File dialog
                    hr = pFileOpen->Show(NULL);

                    if (SUCCEEDED(hr)) {
                        IShellItem* pItem;
                        hr = pFileOpen->GetResult(&pItem);
                        if (SUCCEEDED(hr)) {
                            PWSTR filePath = nullptr;
                            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);

                            if (SUCCEEDED(hr)) {
                                strDirectory = filePath;
                                CoTaskMemFree(filePath);
                            }
                            pItem->Release();
                        }
                    }
                    pFileOpen->Release();
                } else {
                    LOG_ERROR("Failed to create FileOpenDialog: {:#x}", static_cast<unsigned int>(hr));
                    CoUninitialize();
                }
                CoUninitialize();
            }
            return hr;
        }
    } // namespace utils

    void OpenProjectFileDialog() {
        // show native windows file dialog for project file selection
        PWSTR filePath = nullptr;

        COMDLG_FILTERSPEC rgSpec[] =
            {
                {L"LichtFeldStudio Project File", L"*.lfs;*.ls"},
            };

        if (SUCCEEDED(lfs::vis::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path project_path(filePath);
            lfs::core::events::cmd::LoadProject{.path = project_path}.emit();
            LOG_INFO("Loading project file : {}", std::filesystem::path(project_path).string());
        }
    }

    void OpenPlyFileDialog() {
        // show native windows file dialog for PLY file selection
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] =
            {
                {L"Point Cloud", L"*.ply;"},
            };
        if (SUCCEEDED(lfs::vis::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path ply_path(filePath);
            lfs::core::events::cmd::LoadFile{.path = ply_path}.emit();
            LOG_INFO("Loading PLY file : {}", std::filesystem::path(ply_path).string()); // FIXED: Changed from "Loading project file"
        }
    }

    void OpenDatasetFolderDialog() {
        // show native windows file dialog for folder selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(lfs::vis::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path dataset_path(filePath);
            if (std::filesystem::is_directory(dataset_path)) {
                lfs::core::events::cmd::LoadFile{.path = dataset_path, .is_dataset = true}.emit();
                LOG_INFO("Loading dataset : {}", std::filesystem::path(dataset_path).string());
            }
        }
    }

    void SaveProjectFileDialog(bool* p_open) {
        // show native windows file dialog for project directory selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(lfs::vis::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path project_path(filePath);
            lfs::core::events::cmd::SaveProject{project_path}.emit();
            LOG_INFO("Saving project file into : {}", std::filesystem::path(project_path).string());
            *p_open = false;
        }
    }

    namespace utils {
        HRESULT saveFileNative(PWSTR& outPath,
                               COMDLG_FILTERSPEC rgSpec[],
                               UINT cFileTypes,
                               const wchar_t* defaultName) {
            HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(hr)) {
                LOG_ERROR("Failed to initialize COM: {:#x}", static_cast<unsigned int>(hr));
                return hr;
            }

            IFileSaveDialog* pFileSave = nullptr;
            hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL,
                                  IID_IFileSaveDialog, reinterpret_cast<void**>(&pFileSave));

            if (SUCCEEDED(hr)) {
                if (rgSpec != nullptr && cFileTypes > 0) {
                    pFileSave->SetFileTypes(cFileTypes, rgSpec);
                    pFileSave->SetFileTypeIndex(1);
                    pFileSave->SetDefaultExtension(L"ply");
                }

                if (defaultName != nullptr) {
                    pFileSave->SetFileName(defaultName);
                }

                hr = pFileSave->Show(NULL);

                if (SUCCEEDED(hr)) {
                    IShellItem* pItem;
                    hr = pFileSave->GetResult(&pItem);
                    if (SUCCEEDED(hr)) {
                        PWSTR filePath = nullptr;
                        hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);
                        if (SUCCEEDED(hr)) {
                            outPath = filePath;
                        }
                        pItem->Release();
                    }
                }
                pFileSave->Release();
            }
            CoUninitialize();
            return hr;
        }
    } // namespace utils

#endif // WIN32

    namespace {
#ifndef _WIN32
        // Execute dialog command and return trimmed output
        std::string runDialogCommand(const std::string& primary_cmd, const std::string& fallback_cmd) {
            FILE* pipe = popen(primary_cmd.c_str(), "r");
            if (!pipe && !fallback_cmd.empty()) {
                pipe = popen(fallback_cmd.c_str(), "r");
            }
            if (!pipe) return {};

            char buffer[4096];
            std::string result;
            while (fgets(buffer, sizeof(buffer), pipe)) {
                result += buffer;
            }
            pclose(pipe);

            while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                result.pop_back();
            }
            return result;
        }
#endif
    } // namespace

    std::filesystem::path SaveJsonFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"JSON File", L"*.json"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".json") {
                result += ".json";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='JSON files|*.json' "
                                    "--filename='" + defaultName + ".json' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'JSON files (*.json)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty()) return {};

        std::filesystem::path path(result);
        if (path.extension() != ".json") {
            path += ".json";
        }
        return path;
#endif
    }

    std::filesystem::path OpenJsonFileDialog() {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"JSON File", L"*.json"}};

        if (SUCCEEDED(utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --file-filter='JSON files|*.json' 2>/dev/null";
        const std::string fallback = "kdialog --getopenfilename . 'JSON files (*.json)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty()) return {};
        return std::filesystem::path(result);
#endif
    }

    std::filesystem::path SavePlyFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"PLY Point Cloud", L"*.ply"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".ply") {
                result += ".ply";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='PLY files|*.ply' "
                                    "--filename='" + defaultName + ".ply' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'PLY files (*.ply)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty()) return {};

        std::filesystem::path path(result);
        if (path.extension() != ".ply") {
            path += ".ply";
        }
        return path;
#endif
    }

    std::filesystem::path SaveSogFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"SOG File (SuperSplat)", L"*.sog"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".sog") {
                result += ".sog";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='SOG files (SuperSplat)|*.sog' "
                                    "--filename='" + defaultName + ".sog' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'SOG files (*.sog)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty()) return {};

        std::filesystem::path path(result);
        if (path.extension() != ".sog") {
            path += ".sog";
        }
        return path;
#endif
    }

} // namespace lfs::vis::gui